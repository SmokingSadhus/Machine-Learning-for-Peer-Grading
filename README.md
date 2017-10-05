# Machine-Learning-for-Peer-Grading

Proposal for Master’s Independent Study (CSC 630)
                                   Project: A Machine-Learning approach to peer grading 
                                   Student Name: Abhinav Nilesh Medhekar
                                         Unity ID: amedhek
                                         Student ID: 200156110 
                                         Instructor: Dr. Edward Gehringer 

Proposed Idea:
The main goal of this project is to compute the final grade on any assignment submission using peer review grades without any instructor intervention. For doing this, there are two steps 1) Find out how reliable a particular peer review is.  2) Use the reliability rating of each review to find out how much that review should count towards the final grade. In this Semester, I plan to work on step 1. We have a set of around 2000 reviews with staff scores (in the range of 0-100) which I plan to use as a training set. I will train a neural network with significant features extracted from these reviews. The goal of this network will be to classify the review into one of 5 ranges (number of ranges might be changed in order to achieve better results). For e.g. scores of 0 – 20 will be range 1, scores of 20 – 40 will be range 2 and so on. The features I plan to use fall into 3 categories. Category one features are based on the feedback written by the reviewer. Here I am planning to take into account the length of the feedback, the number of repeating phrases/ words in the feedback, and also whether the written description is really a review or just some random text copied from the web. For the last part I am planning to use a technique similar to Spam filtering. Category two features will be based on the scores given to various criteria as part of the review. I will compute how similar the scores are compared to scores by other reviewers for the same submission. If the scores match most of the other scores, it may indicate a reliable review. Also, for a particular submission, how similar are the scores for different criteria for e.g. scores of 5 for all the criteria could indicate a bad review. Category three features will be based on the reviewer’s performance in the course and also his scores on a quiz related to the material he is reviewing. The challenge here would be to convert each of these extracted features into numerical values for training the neural network.
Scheduled Activities:
Extract features belonging to the three mentioned categories from the training set.
Convert the features to numerical values for training the network.
Divide the available labelled data into training and test set and use the training set to train the neural network.
If the network does not show good results on the test set, try using a different neural network or a different classification technique like a decision tree.
Prepare a report with the final results achieved and any suggestions for improvement.
