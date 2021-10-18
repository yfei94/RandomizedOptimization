import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def generate_neural_net_rhc_curves(train_X, test_X, train_Y, test_Y):
    # Creating initial classifier with no hyper parameter tuning
    clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='random_hill_climb', random_state=12)

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("RHC Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("RHC Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyperparameters
    train_scores = []
    test_scores = []
    max_iters_range = []
    
    for max_iters in range(500, 5101, 500):
        max_iters_range.append(max_iters)
        clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='random_hill_climb', max_iters=max_iters, random_state=12)
        clf.fit(train_X, train_Y)
        predict_Y = clf.predict(train_X)
        train_scores.append(accuracy_score(train_Y, predict_Y))
        predict_Y = clf.predict(test_X)
        test_scores.append(accuracy_score(test_Y, predict_Y))

    best_index = np.argmax(test_scores)
    best_max_iters = max_iters_range[best_index]
    print("Best validation score for max number of iterations: ", test_scores[best_index])
    print("Best max number of iterations: ", best_max_iters)

    plt.title("Neural Network Learning curve with RHC")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.plot(max_iters_range, train_scores, label="Training accuracy")
    plt.plot(max_iters_range, test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/sphere_rhc_learning_curve.png')
    plt.clf()

    # Validation curve for num restarts

    train_scores = []
    test_scores = []
    restarts_range = []

    for restarts in range(0, 6):
        restarts_range.append(restarts)
        clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='random_hill_climb', max_iters=4500, random_state=12, restarts=restarts)
        clf.fit(train_X, train_Y)
        predict_Y = clf.predict(train_X)
        train_scores.append(accuracy_score(train_Y, predict_Y))
        predict_Y = clf.predict(test_X)
        test_scores.append(accuracy_score(test_Y, predict_Y))

    best_index = np.argmax(test_scores)
    best_restarts = restarts_range[best_index]
    print("Best validation score for number of restarts: ", test_scores[best_index])
    print("Best number of restarts: ", best_restarts)

    plt.title("Neural Network validation curve with RHC on number of restarts")
    plt.xlabel("Number of restarts")
    plt.ylabel("Accuracy")
    plt.plot(restarts_range, train_scores, label="Training accuracy")
    plt.plot(restarts_range, test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/sphere_rhc_validation_curve_restarts.png')
    plt.clf()

def generate_neural_net_sa_curves(train_X, test_X, train_Y, test_Y):
    # Creating initial classifier with no hyper parameter tuning
    clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='simulated_annealing', random_state=12)

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("SA Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("SA Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyperparameters
    train_scores = []
    test_scores = []
    max_iters_range = []
    
    for max_iters in range(1000, 6101, 500):
        max_iters_range.append(max_iters)
        clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='simulated_annealing', max_iters=max_iters, random_state=12)
        clf.fit(train_X, train_Y)
        predict_Y = clf.predict(train_X)
        train_scores.append(accuracy_score(train_Y, predict_Y))
        predict_Y = clf.predict(test_X)
        test_scores.append(accuracy_score(test_Y, predict_Y))

    best_index = np.argmax(test_scores)
    best_max_iters = max_iters_range[best_index]
    print("Best validation score for max number of iterations: ", test_scores[best_index])
    print("Best max number of iterations: ", best_max_iters)

    plt.title("Neural Network Learning curve with SA")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.plot(max_iters_range, train_scores, label="Training accuracy")
    plt.plot(max_iters_range, test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/sphere_sa_learning_curve.png')
    plt.clf()

    train_scores = []
    test_scores = []
    decay_range = []
    
    for decay in np.arange(0.85, 1, 0.02):
        decay_range.append(decay)
        schedule = mlrose_hiive.GeomDecay(decay=decay)
        clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='simulated_annealing', max_iters=5500, random_state=12, schedule=schedule)
        clf.fit(train_X, train_Y)
        predict_Y = clf.predict(train_X)
        train_scores.append(accuracy_score(train_Y, predict_Y))
        predict_Y = clf.predict(test_X)
        test_scores.append(accuracy_score(test_Y, predict_Y))

    best_index = np.argmax(test_scores)
    best_decay = decay_range[best_index]
    print("Best validation score for decay: ", test_scores[best_index])
    print("Best decay: ", best_decay)

    plt.title("Neural Network Validation curve with SA on decay")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.plot(decay_range, train_scores, label="Training accuracy")
    plt.plot(decay_range, test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/sphere_sa_validation_curve_decay.png')
    plt.clf()

def generate_neural_net_ga_curves(train_X, test_X, train_Y, test_Y):
    # Creating initial classifier with no hyper parameter tuning
    clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='genetic_alg', random_state=12, pop_size=40)

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("GA Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("GA Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyperparameters
    train_scores = []
    test_scores = []
    max_iters_range = []
    
    for max_iters in range(500, 1001, 100):
        print(max_iters)
        max_iters_range.append(max_iters)
        clf = mlrose_hiive.NeuralNetwork([10, 10], algorithm='genetic_alg', max_iters=max_iters, random_state=12, pop_size=40)
        clf.fit(train_X, train_Y)
        predict_Y = clf.predict(train_X)
        train_scores.append(accuracy_score(train_Y, predict_Y))
        predict_Y = clf.predict(test_X)
        test_scores.append(accuracy_score(test_Y, predict_Y))

    best_index = np.argmax(test_scores)
    best_max_iters = max_iters_range[best_index]
    print("Best validation score for max number of iterations: ", test_scores[best_index])
    print("Best max number of iterations: ", best_max_iters)

    plt.title("Neural Network Learning curve with GA")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.plot(max_iters_range, train_scores, label="Training accuracy")
    plt.plot(max_iters_range, test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/sphere_ga_learning_curve.png')
    plt.clf()


if (__name__ == '__main__'):
    features = ['x', 'y', 'z']

    df_sphere = pd.read_csv('data/sphere.csv')

    X = df_sphere[features]
    Y = df_sphere['output']

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=12, test_size=0.2)

    generate_neural_net_rhc_curves(train_X, test_X, train_Y, test_Y)
    generate_neural_net_sa_curves(train_X, test_X, train_Y, test_Y)
    generate_neural_net_ga_curves(train_X, test_X, train_Y, test_Y)