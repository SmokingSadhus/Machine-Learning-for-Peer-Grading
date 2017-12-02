import csv
import re

def clean_str(str):
    str = ''.join(i for i in str if ord(i)<128)
    str = re.sub(',', ' , ', str)
    str = re.sub('\.', ' . ', str)
    str = re.sub('!', ' ! ', str)    
    str = re.sub('[^A-Za-z0-9\s]+', '', str)
    str = re.sub('\s+',' ',str)
    return str.strip().lower()

def get_class_value(score):
    if score <= 50:
        return 0
    elif score > 50 and score <=70:
        return 1
    elif score > 70 and score <=85:
        return 2
    elif score > 85 and score <=95:
        return 3
    elif score > 95:
        return 4


def c_to_int(string):
    try:
        return int(string)
    except ValueError:
        return 0


def process_grades(g):
	g = g.strip("%")
	g = g.split(".")
	return int(g[0])

def load_data_and_labels(file_name):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text = []
    y = []
    with open(file_name, encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #print(row['comments'])
            if row['grade_for_reviewer'] != 'NULL':
                noofcomments = len(row['scores'].split(","))
                if noofcomments != 0 :
                    cleaned_string = clean_str(row['comments'])
                    x_text.append(cleaned_string)
                    y.append(get_class_value(process_grades(row['grade_for_reviewer'])))
    return [x_text,y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
        
[a,b] = load_data_and_labels('DataSet5.csv')
#print (clean_str('Yes, but for the purposes of our class and the review process,'))
