import time
import json
import math
from scipy.interpolate import interp1d
import numpy as np
import copy 
from scipy.spatial import distance

centroids_X = [200, 650, 450, 400, 350, 500, 600, 700, 850, 800, 900, 1000, 850, 750, 950, 1050, 150, 450, 300, 550, 750, 550, 250, 350, 650, 250 ]
centroids_Y = [250, 350, 350, 250, 150, 250, 250, 250, 150, 250, 250, 250 , 350, 350, 150, 150 , 150, 150, 250, 150, 150, 350, 150, 350, 150, 350 ]


# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []

def generate_templates():
    global words, probabilities, template_sample_points_X, template_sample_points_Y
    template_points_X, template_points_Y = [], []
    file = open('words_10000.txt')
    content = file.read()
    file.close()
    content = content.split('\n')
    for line in content:
        line = line.split('\t')
        words.append(line[0])
        probabilities[line[0]] = float(line[2])
        template_points_X.append([])
        template_points_Y.append([])
        for c in line[0]:
            template_points_X[-1].append(centroids_X[ord(c) - 97])
            template_points_Y[-1].append(centroids_Y[ord(c) - 97])
    for i in range(10000):
        X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
        template_sample_points_X.append(X)
        template_sample_points_Y.append(Y)

def display_template():
    global template_sample_points_X
    print (template_sample_points_X)

# To checked distance within threshold
def distance_threshold(p1,p2,threshold):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) 
    if dist < threshold:
        return True
    else:
        return False
# Calculate distance between points
def distance_points(p1,p2):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) 
    return dist    
# Function to plot interp1d curve for x and y, so as to not lose integrity and find equidistant points.
def sample_points(points_X, points_Y):
    dist = np.cumsum(np.sqrt( np.ediff1d(points_X, to_begin=0)**2 + np.ediff1d(points_Y, to_begin=0)**2 ))
    if sum(dist) == 0:
        return [points_X[0] for i in range(100)], [points_Y[0] for i in range(100)]
    dist = dist/dist[-1]
    func_x, func_y = interp1d( dist, points_X ), interp1d( dist, points_Y )
    req_Points = np.linspace(0, 1, 100)
    return func_x(req_Points), func_y(req_Points)
#Samples are generated as per paper.
def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)
    sample_points_X, sample_points_Y =  sample_points(points_X, points_Y)
    return sample_points_X, sample_points_Y


# Pruning is done as per paper.
def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    global words
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    gest_start = (gesture_points_X[0],gesture_points_Y[0])
    gest_end = (gesture_points_X[-1],gesture_points_Y[-1])
    threshold = 50
    # TODO: Do pruning (12 points)
    for i in range(len(template_sample_points_X)):
        x = distance_threshold(gest_start,(template_sample_points_X[i][0],template_sample_points_Y[i][0]),threshold)
        if not x:
            continue
        y = distance_threshold(gest_end,(template_sample_points_X[i][-1],template_sample_points_Y[i][-1]),threshold)
        if not y:
            continue
        valid_template_sample_points_X.append(template_sample_points_X[i])
        valid_template_sample_points_Y.append(template_sample_points_Y[i])
        valid_words.append(words[i])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y
# Function to calculate bounding box coordinates.
def bou_Box(points_X , points_Y):
    return (min(points_X), min(points_Y)), (max(points_X), max(points_Y))
# Subtract Centroid with each point
def centr_subtract(points_X , points_Y, C):
    points_X = np.array(points_X)
    points_Y = np.array(points_Y)
    sub_X = np.array(C[0]*100)
    sub_Y = np.array(C[1]*100)
    return np.subtract(points_X,sub_X),np.subtract(points_Y,sub_Y)
# Scale each point after calculating scaling factor
def point_scale(points_X , points_Y, K, L):
    S = L/K
    sca_list = np.array([S]*100)
    return np.multiply(points_X,sca_list),np.multiply(points_Y,sca_list)
# To normalize all pair or gestures/templates
def normalize_pair(p1,p2,L):
    bound_1, bound_2 = bou_Box(p1,p2)
    width = bound_2[0] - bound_1[0] 
    height = bound_2[1] - bound_1[1]
    centroid =  ((bound_1[0] + bound_2[0])/2,(bound_1[1] + bound_2[1])/2)
    p1,p2 = centr_subtract(p1,p2,centroid)
    if width > height and width != 0:
        p1,p2 = point_scale(p1,p2,width,L)
    if width < height and height != 0:
        p1,p2 = point_scale(p1,p2,height,L)
    return p1,p2
# 2-D Normalisation done by calculating centroid, making it the point of origin and scaling
def perform_shape_normalisation(uX,uY,tX,tY,L):
    uX,uY = normalize_pair(uX,uY,L)
    for i in range(len(tX)):
        tX[i],tY[i] = normalize_pair(tX[i],tY[i],L)
    return uX,uY,tX,tY

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = np.zeros(len(valid_template_sample_points_X))
    # TODO: Set your own L
    L = 1
    # TODO: Calculate shape scores (12 points)
    g_points_x,g_points_y=gesture_sample_points_X, gesture_sample_points_Y
    t_points_x = copy.deepcopy(valid_template_sample_points_X)  
    t_points_y = copy.deepcopy(valid_template_sample_points_Y)
    g_points_x,g_points_y,t_points_x,t_points_y = perform_shape_normalisation(g_points_x,g_points_y,t_points_x,t_points_y,L)
    score = 0
    for i in range(len(valid_template_sample_points_X)):
        score = 0
        for j in range(100):
            score = score + distance_points((g_points_x[j],g_points_y[j]),(t_points_x[i][j],t_points_y[i][j]))/100
        shape_scores[i] = (score)
    return shape_scores
# Function to calculate the D(U,T)/D(T,U) values
def big_d_calc(tX,tY,uX,uY,R):
    tX,tY,uX,uY = np.array(tX),np.array(tY),np.array(uX),np.array(uY)
    good = [np.array((tX[i],tY[i])) for i in range(len(uX))]
    poi_2 = np.array((uX,uY)).T
    sum1 = distance.cdist(good, poi_2, 'euclidean')
    sum1 = np.subtract(sum1,R)
    sum1 = np.sum(sum1,axis =1)
    return np.sum(sum1[sum1 > 0 ])
# Else condition, calculating the score directly
def location_score_calc(gX,gY,tX,tY):
    alpha = np.array([0.0196078431372549, 0.019215686274509803, 0.018823529411764704, 0.01843137254901961, 0.01803921568627451, 0.01764705882352941, 0.017254901960784313, 0.016862745098039214, 0.01647058823529412, 0.01607843137254902, 0.01568627450980392, 0.015294117647058824, 0.014901960784313726, 0.014509803921568627, 0.01411764705882353, 0.013725490196078431, 0.013333333333333334, 0.012941176470588235, 0.012549019607843137, 0.01215686274509804, 0.011764705882352941, 0.011372549019607842, 0.010980392156862745, 0.010588235294117647, 0.01019607843137255, 0.00980392156862745, 0.009411764705882352, 0.009019607843137255, 0.008627450980392156, 0.00823529411764706, 0.00784313725490196, 0.007450980392156863, 0.007058823529411765, 0.006666666666666667, 0.006274509803921568, 0.0058823529411764705, 0.005490196078431373, 0.005098039215686275, 0.004705882352941176, 0.004313725490196078, 0.00392156862745098, 0.0035294117647058825, 0.003137254901960784, 0.0027450980392156863, 0.002352941176470588, 0.00196078431372549, 0.001568627450980392, 0.001176470588235294, 0.000784313725490196, 0.000392156862745098, 0.000392156862745098, 0.000784313725490196, 0.001176470588235294, 0.001568627450980392, 0.00196078431372549, 0.002352941176470588, 0.0027450980392156863, 0.003137254901960784, 0.0035294117647058825, 0.00392156862745098, 0.004313725490196078, 0.004705882352941176, 0.005098039215686275, 0.005490196078431373, 0.0058823529411764705, 0.006274509803921568, 0.006666666666666667, 0.007058823529411765, 0.007450980392156863, 0.00784313725490196, 0.00823529411764706, 0.008627450980392156, 0.009019607843137255, 0.009411764705882352, 0.00980392156862745, 0.01019607843137255, 0.010588235294117647, 0.010980392156862745, 0.011372549019607842, 0.011764705882352941, 0.01215686274509804, 0.012549019607843137, 0.012941176470588235, 0.013333333333333334, 0.013725490196078431, 0.01411764705882353, 0.014509803921568627, 0.014901960784313726, 0.015294117647058824, 0.01568627450980392, 0.01607843137254902, 0.01647058823529412, 0.016862745098039214, 0.017254901960784313, 0.01764705882352941, 0.01803921568627451, 0.01843137254901961, 0.018823529411764704, 0.019215686274509803, 0.0196078431372549])
    template_X = np.array((gX,gY)).T
    template_Y = np.array((tX,tY)).T
    template_Z = np.subtract(template_X,template_Y)
    template_Z = np.square(template_Z)
    template_Z = np.sum(template_Z,axis=1)
    template_Z = np.sqrt(template_Z)
    return np.sum(np.multiply(template_Z,alpha))
# Location score done as per paper
def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = np.zeros(len(valid_template_sample_points_X))
    radius = 15
    # TODO: Calculate location scores (12 points)
    for i in range(len(valid_template_sample_points_X)):
        x = big_d_calc(gesture_sample_points_X, gesture_sample_points_Y,valid_template_sample_points_X[i], valid_template_sample_points_Y[i],radius)
        y = big_d_calc(valid_template_sample_points_X[i], valid_template_sample_points_Y[i],gesture_sample_points_X, gesture_sample_points_Y,radius)
        if not (x + y == 0):
            location_scores[i] = (location_score_calc(gesture_sample_points_X, gesture_sample_points_Y,valid_template_sample_points_X[i], valid_template_sample_points_Y[i]))
    return location_scores
# Location coef set to 0.7 as it is more important to find correct match
def get_integration_scores(shape_scores, location_scores):
    shape_coef = 0.3
    location_coef = 0.7
    return (shape_scores * shape_coef) + (location_coef * location_scores)
# Function to return indices of minimum scores
def locate_min(list_):
    min_ = min(list_)
    return [i for i, elem in enumerate(list_) 
                      if min_ == elem]

def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = ''
    # TODO: Set your own range.
    # n = 3
    # TODO: Get the best word (12 points)
    if len(integration_scores) == 0:
        return 'No gesture found'
    x = locate_min(integration_scores)
    for i in range(len(valid_words)):
        print(valid_words[i],integration_scores[i])
    for i in x:
        best_word = best_word +' ' + valid_words[i]
    return best_word

def predict_word(gesture_points_X, gesture_points_Y):
    global template_sample_points_X, template_sample_points_Y

    # print(gesture_points_X,gesture_points_Y)
    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    return best_word