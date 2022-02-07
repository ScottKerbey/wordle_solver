# Databricks notebook source
# MAGIC %md
# MAGIC # Analysis of word information gain for the game wordle
# MAGIC Which word gives the most information on the first guess?

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
import databricks.koalas as ks
import pandas as pd

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

dict_df = spark.sql("SELECT * FROM wordle.dictionary_txt")

# COMMAND ----------

dictionary = dict_df.toPandas().transpose().values[0].tolist()

# COMMAND ----------

dictionary

# COMMAND ----------

# # Load the dictionary from file as a list
# with open("/dbfs/FileStore/tables/dictionary.txt",'r') as f:
#     dictionary = [line.strip() for line in f]

# COMMAND ----------

empty_list = [[dictionary[0],i] for i in dictionary]

# COMMAND ----------

len(empty_list), len(dictionary)

# COMMAND ----------

# create an emtpy spark dataframe with the dictionary as the column names
matrix_df = spark.createDataFrame([empty_list], schema=dictionary)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE wordle.dict_matrix

# COMMAND ----------

matrix_df.write.saveAsTable("wordle.dict_matrix")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM wordle.dict_matrix

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE wordle.dict_matrix

# COMMAND ----------

spark.sql('''
SELECT *
FROM wordle.dict_matrix
''').select(dictionary[:10]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Rules for the Game

# COMMAND ----------

# MAGIC %md
# MAGIC ### Guess the word

# COMMAND ----------

# return an 5-array with 'c', 'p' or 'a' for correct, present or absent
# if there is a double letter in the guess but only one in the answer then the second letter is 'a'
def guess_word(guess, answer):
    # create and empty info array
    info = ['a'] * 5

    # count the number of each letter in the answer
    letter_count = {}
    for letter in answer:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1
    
    # find all the correct letters
    for index, letter in enumerate(guess):
        if answer[index] == letter:
            info[index] = 'c' # correct
            # remove the letter from the letter_count
            letter_count[letter] -= 1

    # find all the present letters
    for index, letter in enumerate(guess):
        if answer[index] != letter:
            if letter in answer:
                # check if the letter count is greater than 0
                if letter_count[letter] > 0:
                    info[index] = 'p' # present
                    letter_count[letter] -= 1
    return info

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display word guesses

# COMMAND ----------

from IPython.display import HTML, display

def display_table(data):
    html = '<table style="padding: 10px;border: 3px solid goldenrod; ">'
    for row in data:
        html += '<tr style="height:32px">'
        for field in row:
            colour = 'green' if field[1] == 'c' else 'yellow' if field[1] == 'p' else 'grey'
            html += '<td style="text-align:center;width:32px;background-color:%s;"><h4>%s</h4></td>'%(colour, field[0])
        html += "</tr>"
    html += "</table>"
    display(HTML(html))

# COMMAND ----------

def guess_pretty(guess, answer):
    data = [list(zip(list(guess),guess_word(guess, answer)))]
    display_table(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eliminate the word if it doesnt match the guess and info

# COMMAND ----------

# return whether a word is eliminated based on the info
# this is missing the case where a letter is present but it is in the word index
def eliminate(guess, info, word):
    # count the number of each letter in the word
    letter_count = {}
    for letter in word:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1

    # check if thhe correct letters match
    for index, letter in enumerate(guess):
        if info[index] == 'c':
            if letter != word[index]:
                return True
            # remove the letter from the letter_count
            letter_count[letter] -= 1

    # check all the present letters
    for index, letter in enumerate(guess):
        if info[index] == 'p':
            if letter not in word or letter_count[letter] == 0:
                return True
            if letter == word[index]:
                return True
            # remove the letter from the letter_count
            letter_count[letter] -= 1

    # check all the absent letters
    for index, letter in enumerate(guess):
        if info[index] == 'a':
            if letter in word and letter_count[letter] > 0:
                return True

    return False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reduce the dictionary to remaining words

# COMMAND ----------

# count the number of words eliminated in the dictionary
def dict_reduce(guess, answer, dictionary):
    info = guess_word(guess, answer)
    reduced_dict = [word for word in dictionary if not eliminate(guess, info, word)]
    return reduced_dict

# COMMAND ----------

def dict_red(guess, answer):
    return dict_reduce(guess, answer, dictionary)

# COMMAND ----------

# convert dict_red_mini to at udf
udf_dict_red = udf(lambda z: dict_red(z[0], z[1]), ArrayType(StringType()))

# COMMAND ----------

guess = 'clams'
info = ['a', 'c', 'p', 'a', 'a']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process all guess and answer pairs

# COMMAND ----------

pmatrix_df = matrix_df
col = dictionary[0]
pmatrix_df = pmatrix_df.withColumn(col, udf_dict_red(col))

# COMMAND ----------

pmatrix_df = matrix_df
for col in dictionary:
    pmatrix_df = pmatrix_df.withColumn(col, udf_dict_red(col))
pmatrix_df.select(dictionary[:10]).show()

# COMMAND ----------

test_list = list(zip(dictionary, list(zip(dictionary, [dictionary[0]]*len(dictionary)))))
test_list

# COMMAND ----------

test_df = spark.createDataFrame(test_list, schema=['guess', dictionary[0]])

# COMMAND ----------

col = dictionary[0]
test_df = test_df.withColumn(col, udf_dict_red(col))

# COMMAND ----------

test_df.show()

# COMMAND ----------

test_df.createOrReplaceTempView('new_col')

# COMMAND ----------

schema = StructType([StructField("guess_word",StringType(),True),] + [StructField(word,ArrayType(StringType()), True) for word in dictionary])

# COMMAND ----------

schema

# COMMAND ----------

emptyRDD = spark.sparkContext.emptyRDD()

reduced_dict_df = spark.createDataFrame(emptyRDD,schema)

# COMMAND ----------

reduced_dict_df.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE wordle.reduced_dict

# COMMAND ----------

reduced_dict_df.write.saveAsTable("wordle.reduced_dict")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE wordle.reduced_dict

# COMMAND ----------

guess_col_df = spark.createDataFrame([[word] for word in dictionary]).withColumnRenamed('_1', 'guess_word')

# COMMAND ----------

guess_col_df.show()

# COMMAND ----------

guess_col_df.createOrReplaceTempView('guess_col')

# COMMAND ----------

INSERT INTO tableName SELECT t.* FROM (SELECT value1, value2, ... ) t;

# COMMAND ----------

", ".join(['a','b','c'])

# COMMAND ----------

spark.sql(f'''
INSERT INTO wordle.reduced_dict
SELECT guess_word, {", ".join(['null']*len(dictionary))}
FROM guess_col
''')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM wordle.reduced_dict

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT rd.guess_word, new_col.aahed
# MAGIC FROM wordle.reduced_dict rd
# MAGIC INNER JOIN new_col ON rd.guess_word = new_col.guess

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO wordle.reduced_dict (aahed)
# MAGIC SELECT new_col.aahed
# MAGIC FROM new_col
# MAGIC WHERE wordle.reduced_dict = new_col.guess

# COMMAND ----------


