#_______________Binary Logistic Regression Model_______________

#Import system modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import math

#Clock start time to evaluate model performance
t0 = time.time()

#_______________Import and clean training data_______________

ExoplanetData = pd.read_csv("ExoplanetData.csv")
TrainingDataArray = ((pd.DataFrame([ExoplanetData.fpl_smax, ExoplanetData.fpl_bmasse, ExoplanetData.fpl_rade, ExoplanetData.fpl_dens]).transpose()).dropna()).to_numpy()

#Filter anomalous data (planets with unexplained density values)
Filter_indices = []
TrainingData_length = TrainingDataArray.shape[0] #Store number of data records in training data

for i in range(0, TrainingData_length):
    if TrainingDataArray[i, 3] > 10:
        Filter_indices.append(i)

TrainingDataArray = np.delete(TrainingDataArray, Filter_indices, 0)
TrainingData_length = TrainingDataArray.shape[0] #Update filtered number of data records in training data

#_______________Classify training examples as gas giants or not_______________

#Classification mass and radius criteria
Gasgiantmass = 15 #Minimum gas giant mass (Me)
Gasgiantradius = 3 #Minimum gas giant radius (Re)

#Restricted density classification criteria appropriate for gas giant planets
Gasgiantdensity = 2 #Maximum gas giant density (kg/m3)

#No density classification criteria
#Gasgiantdensity = 10 #Maximum gas giant density (kg/m3)

Gasgiant = np.zeros((TrainingData_length, 1))

#Gas giant = 1, non-gas giant = 0
for i in range(0, TrainingData_length):
    if TrainingDataArray[i, 1] >= Gasgiantmass and TrainingDataArray[i, 2] >= Gasgiantradius and TrainingDataArray[i, 3] <= Gasgiantdensity: 
        Gasgiant[i] = 1
    else:
        Gasgiant[i] = 0
  
TrainingDataArray = np.append(TrainingDataArray, Gasgiant, axis=1)

#Extract binary training data variables to new array
TrainingData = np.zeros((TrainingData_length, 2))
TrainingData[:,0] = TrainingDataArray[:,0] #Semi-major axis values
TrainingData[:,1] = TrainingDataArray[:,4] #Gas giant classification

#_______________Plot training data for visualization_______________

Training_data_view = 1

if Training_data_view == 1:
    plt.figure(1)
    plt.plot(TrainingDataArray[:,0], TrainingDataArray[:,1], 'ko')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Semi-major Axis (AU)')
    plt.ylabel('Planet Mass (Me)')

    plt.figure(2)
    plt.plot(TrainingDataArray[:,0], TrainingDataArray[:,4], 'ko')
    plt.xscale('log')
    plt.xlabel('Orbit Distance (AU)')
    plt.ylabel('Gas Giant (class = 1)')
    plt.show()

#_______________Toggle optimization method and code features_______________
    
batch_gradient = 0 #Toggle batch gradient descent optimization (1 = on, 0 = off), toggle only 1 optimization method
stochastic_gradient = 0 #Toggle stochastic gradient descent optimization (1 = on, 0 = off)
mini_batch = 1 #Toggle mini batch gradient descent optimization (1 = on, 0 = off)
iterative_plotting = 0 #Toggle iterative plotting to view gradient descent history
MC_cross_validation = 0 #Toggle for Monte Carlo cross-validation (greatly increases run time), else a simple holdout validation will be performed
MC_validation_iterations = 25 #Number of model runs to perform for Monte Carlo cross-validation
final_plots = 1 #Toggle to generate final plots to visualize model results (1 = on, 0 = off)

#Set validation iteration counter 
if MC_cross_validation == 1:
    cross_validation_iterations = MC_validation_iterations
else:
    cross_validation_iterations = 1
    
cross_validation_score = np.zeros(cross_validation_iterations) #Pre-allocate cross validation score array

#_______________Define learning algorithm parameters_______________

learning_rate = 0.0000005 #Gradient descent learning rate
max_iterations = 25000 #Maximum number of gradient descent iterations
converge_tolerance = 0.000001 #Tolerance for gradient descent convergence
mini_batch_size = 10 #Size of batches for use in mini batch gradient descent
validation_percentage = 10 #Percentage of training data to be removed and used for cross validation
validation_examples = int((validation_percentage/100)*TrainingData_length) #Store number of validation examples
training_examples = TrainingData_length - validation_examples #Store number of training data examples
eps = 1e-15 #Adjustment value for log-loss calculation
iteration_count = 0 #Iteration progress counter

#Set iterative plotting counters
plot_iteration_counter = 0
plot_save_counter = 0

#_______________Train and validate logistic regression model_______________

for m in range(0, cross_validation_iterations):
    
    #Randomize and split data to create cross-validation and training data arrays
    np.random.shuffle(TrainingData)

    X_validation = TrainingData[0:validation_examples, 0]
    Y_validation = TrainingData[0:validation_examples, 1]

    #Re-order data and extract training subset
    TrainingSet = TrainingData[validation_examples:len(TrainingData), :]
    TrainingSet = TrainingSet[np.argsort(TrainingSet[:, 0])]
    X = TrainingSet[:,0]
    Y = TrainingSet[:,1]

    #Define classification function and logistic regression parameters
    h = np.zeros((training_examples, 1))
    theta = np.zeros(2)
    theta_convergence = np.zeros((1, 2))
    log_loss = 0

    #Fit classification function through gradient descent optimization
    for n in range(0, max_iterations):

        #Store current parameter values to check convergence
        converge_sum = sum(abs(theta))

        #Compute log loss/cross-entropy
        iteration_loss = 0
        for i in range(0, training_examples):
            z = theta[0] + (theta[1]*X[i])       
            h[i] = 1/(1 + np.exp(-z))
            h_loss = max(min(h[i], 1 - eps), eps)
            iteration_loss += (-Y[i]*np.log(h_loss)) - ((1 - Y[i])*np.log(1 - h_loss))

        #Batch gradient descent
        if batch_gradient == 1:
            errsum1 = 0
            errsum2 = 0
            for i in range(0, training_examples):
                errsum1 += (Y[i] - h[i])
                errsum2 += (Y[i] - h[i])*X[i]

            theta[0] += learning_rate*errsum1
            theta[1] += learning_rate*errsum2

        #Stochastic gradient descent
        if stochastic_gradient == 1:
            for i in range(0, training_examples):
                theta[0] += learning_rate*(Y[i] - h[i])
                theta[1] += learning_rate*(Y[i] - h[i])*X[i]

        #Mini-batch gradient descent
        if mini_batch == 1:
            mini_counter = 0

            for i in range(0, math.floor(int(training_examples/mini_batch_size))):  
                errsum1 = 0
                errsum2 = 0
            
                for j in range(mini_counter, mini_counter+mini_batch_size):
                    errsum1 += (Y[j] - h[j])
                    errsum2 += (Y[j] - h[j])*X[j]
    
                theta[0] += learning_rate*errsum1
                theta[1] += learning_rate*errsum2
            
                mini_counter += mini_batch_size

            if (training_examples - (math.floor(int(training_examples/mini_batch_size))*mini_batch_size)) > 0:
                errsum1 = 0
                errsum2 = 0
                
                for j in range(mini_counter, training_examples):
                    errsum1 += (Y[j] - h[j])
                    errsum2 += (Y[j] - h[j])*X[j]
                    
                theta[0] += learning_rate*errsum1
                theta[1] += learning_rate*errsum2
    
        #Store logistic regression parameters to evaluate convergence
        theta_convergence = np.append(theta_convergence, theta.reshape(1,2), axis=0)
        log_loss = np.append(log_loss, (iteration_loss/training_examples))

        #Create and save iterative plots to visualize optimization history
        if iterative_plotting == 1:
            plot_iteration_counter += 1
            if n==0 or plot_iteration_counter > max_iterations/200:
                plot_iteration_counter = 0
                plot_save_counter += 1

                #Plot logistic regression model fits to training data
                plt.clf()
                plt.figure(3)
                plt.plot(X,Y, 'ko', label='Training data')
                plt.plot(X_validation, Y_validation, 'ro', label='Validation data')
                plt.plot(X, h, 'b-', linewidth = 3, label='Logistic regression probability fit')
                plt.xscale('log')
                plt.legend(loc=2)
                plt.xlabel('Semi-major axis (AU)', fontweight='bold')
                plt.ylabel('Probability of gas giant (class = 1)', fontweight='bold')
                plt.xlim([1e-3, 1e4])
                plt.title('Iteration #:'+str(n), fontweight='bold')
                plt.savefig('Plots/Figa'+str(plot_save_counter)+'.png')

                #Plot logistic regression parameters vs. iteration # to visualize convergence
                plt.clf()
                plt.figure(4)
                plt.scatter(theta_convergence[:,0], theta_convergence[:,1], c=np.arange(0, theta_convergence.shape[0]), cmap=cm.jet)
                plt.clim(0, max_iterations)
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                cbar = plt.colorbar()
                cbar.set_label(label='Iteration #', fontweight='bold')
                plt.xlabel('Regression intercept', fontweight='bold')
                plt.ylabel('Regression coefficient', fontweight='bold')
                plt.savefig('Plots/Figb'+str(plot_save_counter)+'.png')

                #Plot logistic regression log loss history to evaluate chosen learning rate value
                plt.clf()
                plt.figure(5)
                plt.plot(log_loss[1:-1], 'k-', linewidth=3)
                plt.xlabel('Iteration #', fontweight='bold')
                plt.ylabel('Log loss/Cross-entropy', fontweight='bold')
                plt.xlim([0, max_iterations])
                plt.ylim([0, 10])
                #plt.gca().set_ylim(bottom=0)
                plt.savefig('Plots/Figc'+str(plot_save_counter)+'.png')

        #Break if convergence tolerance met
        if abs(sum(abs(theta)) - converge_sum) <= converge_tolerance:
            break

        #Print iteration progress:
        iteration_count += 1
        if iteration_count >= max_iterations/10:
            print('Percent of maximum iterations complete:', round(n/max_iterations*100), '%')
            iteration_count = 0

    #Evaluate model performance through cross-validation
    h_validation = np.zeros((validation_examples, 1))
    Y_validation_predicted = np.zeros((validation_examples, 1))
    validation_score = 0

    for i in range(0, validation_examples):
        z = theta[0] + (theta[1]*X_validation[i])       
        h_validation[i] = 1/(1+np.exp(-z))
        if h_validation[i] >= 0.5:
            Y_validation_predicted[i] = 1

        if Y_validation_predicted[i] == Y_validation[i]:
            validation_score += 1

    cross_validation_score[m] = validation_score/validation_examples

    #Print progress
    print('Total code progress:', round(((m+1)/cross_validation_iterations)*100), '%')

#Clock completed time and output total algorithm time
t1 = time.time()
total_time = t1-t0
print('Total time (seconds):', round(total_time,2))
    
#_______________Apply scikit-learn logistic regression for comparison_______________

#Reshape training data for scikit-learn function input
X = X.reshape(-1, 1)

#Define scikit-learn model and parameters
skmodel=LogisticRegression(C=100, solver='lbfgs', tol=converge_tolerance)

#Fit scikit-learn model to training data and store results
skfit=skmodel.fit(X, Y)
skcoef = skmodel.coef_[0]
skintercept = skmodel.intercept_

#Compute classification sigmoid for plot comparison
skh = np.zeros((training_examples,1))
for i in range(0, training_examples):
    z = skintercept + skcoef*X[i]       
    skh[i] = 1/(1 + np.exp(-z))

#_______________Plot and evaluate data_______________

#Print final logistic regression parameters from both models for comparison
print('Model parameters:', theta)
print('Scikit-learn model parameters:', skintercept, skcoef)

#Print and save model cross-validation performance
print('Model validation score:', np.mean(cross_validation_score))
if MC_cross_validation == 1:
    np.savetxt("cross_validation_scores.csv", cross_validation_score, delimiter=",")

if final_plots == 1:

    #Plot logistic regression model fits to training data
    plt.figure(6)
    plt.plot(X, Y, 'ko', label='Training data')
    plt.plot(X_validation, Y_validation, 'ro', label='Validation data')
    plt.plot(X, skh, 'r-', linewidth = 3, label='Scikit-learn model')
    plt.plot(X, h, 'b-', linewidth = 3, label='Logistic regression probability fit')
    plt.plot([0.387,0.387],[0,1], 'k--', label='Mercury semi-major axis')
    plt.xscale('log')
    plt.legend(loc=1)
    plt.xlabel('Semi-major axis (AU)', fontweight='bold')
    plt.ylabel('Probability of gas giant (class = 1)', fontweight='bold')
    plt.xlim([1e-3, 1e4])
    plt.savefig('Fig6.png')

    #Plot logistic regression parameters vs. iteration # to visualize convergence
    plt.figure(7)
    plt.scatter(theta_convergence[0:-1:int(theta_convergence.shape[0]/100),0], theta_convergence[0:-1:int(theta_convergence.shape[0]/100),1], c=np.arange(0,theta_convergence[0:-1:int(theta_convergence.shape[0]/100),0].shape[0]),cmap=cm.jet)
    cbar = plt.colorbar()
    cbar.set_label(label='Iteration #', fontweight='bold')
    cbar_ticks=[0, 25, 50, 75, 99]
    cbar_labels = [i * int(theta_convergence.shape[0]/100) for i in cbar_ticks]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    plt.xlabel('Regression intercept', fontweight='bold')
    plt.ylabel('Regression coefficient', fontweight='bold')
    plt.ylim([-0.01,0.01])
    plt.savefig('Fig7.png')

    #Plot logistic regression log loss history to evaluate chosen learning rate value
    log_loss = np.delete(log_loss, 0)
    plt.figure(8)
    plt.plot(log_loss, 'k-', linewidth=3)
    plt.xlabel('Iteration #', fontweight='bold')
    plt.ylabel('Log loss/Cross-entropy', fontweight='bold')
    plt.xlim([0, max_iterations])
    plt.savefig('Fig8.png')
    plt.show()
