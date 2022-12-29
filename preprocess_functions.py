import tensorflow as tf
from spacy.lang.en import English


def get_lines(filename):
  """
  Reads a text filename and returns the lines of text as a list.

  Args:
    filename: a string containing the target filepath

  Returns:
    A list of strings with one string per line from the target filename
  """

  with open(filename, "r") as f:
    return f.readlines()

def txt_to_dicts(filename):
  """
  Returns a list of dicts of abstract line data.
  Takes in filename, reads it and sorts through each line, extracting things
  like the target label, text of the sentence, how many sentences are in the 
  abstract and what sentence number the target line is.
  """

  input_lines = get_lines(filename)

  abstract_lines = ""
  abstract_samples = []

  for line in input_lines:
    if line.startswith("###"):
      abstract_id = line
      abstract_lines = ""
    
    elif line.isspace():
      abstract_line_split = abstract_lines.splitlines()
      
      for k, abstract_line in enumerate(abstract_line_split):
        line_data = {}
        target_txt_split = abstract_line.split("\t")

        line_data["target"] = target_txt_split[0]
        line_data["text"] = target_txt_split[1].lower()
        line_data["line_number"] = k
        line_data["total_lines"] = len(abstract_line_split) - 1

        abstract_samples.append(line_data)

    else: 
      abstract_lines += line

  return abstract_samples

def split_chars(text):
  return " ".join(list(text))

# get the abstract sentences
def get_abstract_sentences(abstract):
  nlp = English()

  nlp.add_pipe("sentencizer")

  doc = nlp(abstract)

  return [str(sent) for sent in list(doc.sents)]

# get texts and line info and put into list of dicts
def get_abstract_txt_and_lines(abstract):
  abstract_lines = get_abstract_sentences(abstract)

  # get total number of lines
  total_lines_in_sample = len(abstract_lines)

  # go through each line and extract features
  sample_lines = []
  for i, line in enumerate(abstract_lines):
    sample_dict = {}
    sample_dict["text"] = str(line)
    sample_dict["line_number"] = i
    sample_dict["total_lines"] = total_lines_in_sample - 1
    sample_lines.append(sample_dict)

  return sample_lines

# one hot encode the line_numbers
def one_hot_lines(abstract, depth):
  sample_lines = get_abstract_txt_and_lines(abstract)

  # get all line_number values from abstract
  test_abstract_line_numbers = [line["line_number"] for line in sample_lines]

  return tf.one_hot(test_abstract_line_numbers, depth)

# get abstract characters
def get_abstract_chars(abstract):
  return [split_chars(sentence) for sentence in get_abstract_sentences(abstract)]

def make_pred_and_show(loaded_model, abstract):
  test_abstract_line_numbers_one_hot = one_hot_lines(abstract, depth=15)
  test_abstract_total_lines_one_hot = one_hot_lines(abstract, depth=20)
  abstract_lines = get_abstract_sentences(abstract)
  abstract_chars = get_abstract_chars(abstract)

  abstract_preds = tf.argmax(loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                    test_abstract_total_lines_one_hot,
                                                    tf.constant(abstract_lines),
                                                    tf.constant(abstract_chars)),
                                                  verbose=0),
                                            axis=1)
  
  classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
  test_abstract_pred_classes = [classes[i] for i in abstract_preds]

  for i, line in enumerate(abstract_lines):
    print(f"{test_abstract_pred_classes[i]}: {line}")