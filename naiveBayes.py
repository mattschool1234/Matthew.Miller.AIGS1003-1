import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
 

    best_accuracy = -1  
    best_params = None  

    
    common_prior = util.Counter()  
    common_conditional_prob = util.Counter()  
    common_counts = util.Counter()  

    for datum, label in zip(trainingData, trainingLabels):
        common_prior[label] += 1
        for feat, value in datum.items():
            common_counts[(feat, label)] += 1
            if value > 0:  
                common_conditional_prob[(feat, label)] += 1

    for k in kgrid:  
        prior = common_prior.copy()
        conditional_prob = common_conditional_prob.copy()
        counts = common_counts.copy()

        # Apply Laplace 
        for label in self.legalLabels:
            for feat in self.features:
                conditional_prob[(feat, label)] += k
                counts[(feat, label)] += 2 * k  

        # stabilize probabilities make them add up to 1
        prior.normalize()
        for x, count in conditional_prob.items():
            conditional_prob[x] = count / counts[x]

        self.prior = prior
        self.conditionalProb = conditional_prob

        # test performance 
        predictions = self.classify(validationData)
        accuracy = sum(predictions[i] == validationLabels[i] for i in range(len(validationLabels))) / len(validationLabels)

        print(f"Performance validation set for k={k:.3f}: {accuracy * 100:.1f}%")
        if accuracy > best_accuracy:
            best_params = (prior, conditional_prob, k)
            best_accuracy = accuracy

        
        self.prior, self.conditionalProb, self.k = best_params
   
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] 
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
        
  def calculateLogJointProbabilities(self, datum):
  
    
    logJoint = util.Counter()

   
    for label in self.legalLabels:
        logJoint[label] = math.log(self.prior[label])
        for feat, value in datum.items():
            conditional_prob = self.conditionalProb[(feat, label)]
            logJoint[label] += value * math.log(conditional_prob) + (1 - value) * math.log(1 - conditional_prob)

   
    
    return logJoint
  
    def findHighOddsFeatures(self, label1, label2):
       
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        for feat in self.features:
           # Formula for odds ratio: P(feature=1 | label1) / P(feature=1 | label2)
            odds_ratio = (
                self.conditionalProb[(feat, label1)] / self.conditionalProb[(feat, label2)]
            )
            featuresOdds.append((odds_ratio, feat))

        # Sort and take the top 100
        featuresOdds.sort(reverse=True)
        featuresOdds = [feat for _, feat in featuresOdds[:100]]    
        #util.raiseNotDefined()

        return featuresOdds
