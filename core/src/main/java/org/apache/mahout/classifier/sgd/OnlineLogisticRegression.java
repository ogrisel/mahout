/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sgd;

import java.util.Iterator;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

/**
 * Generic definition of a 1 of n logistic regression classifier that returns probabilities in
 * response to a feature vector.  This classifier uses 1 of n-1 coding.
 * <p/>
 * TODO: implement per coefficient annealing schedule TODO: implement symbolic input with string,
 * overall cooccurrence and n-gram hash encoding TODO: implement reporter system to monitor
 * progress
 */
public class OnlineLogisticRegression {
  // coefficients for the classification
  private final Matrix beta;

  // number of categories we are classifying.  This should the number of rows of beta plus one.
  private final int numCategories;

  // information about how long since coefficient rows were updated
  private int step = 0;
  private final Vector updateSteps;

  // these next two control decayFactor^steps exponential type of annealing
  // learning rate and decay factor
  private double mu_0 = 1;
  private double decayFactor = 1 - 1e-3;


  // these next two control 1/steps^forget type annealing
  private int stepOffset = 10;
  // -1 equals even weighting of all examples, 0 means only use exponential annealing
  private double forgettingExponent = -0.5;

  // prior and weight
  private double lambda = 0.1;
  private PriorFunction prior;

  private boolean sealed = false;

  public OnlineLogisticRegression(int numCategories, int numFeatures, PriorFunction prior) {
    this.numCategories = numCategories;
    this.prior = prior;

    updateSteps = new DenseVector(numFeatures);
    beta = new DenseMatrix(numCategories - 1, numFeatures);
  }

  /**
   * Chainable configuration option.
   *
   * @param alpha New value of decayFactor, the exponential decay rate for the learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression alpha(double alpha) {
    this.decayFactor = alpha;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param learningRate New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression learningRate(double learningRate) {
    this.mu_0 = learningRate;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param lambda New value of lambda, the weighting factor for the prior distribution.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression lambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  /**
   * Returns n-1 probabilities, one for each category but the last.  The probability of the n-th
   * category is 1 - sum(this result).
   *
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each of the first n-1 categories.
   */
  public Vector classify(Vector instance) {
    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    Vector v = beta.times(instance).assign(Functions.exp);
    double sum = 1 + v.norm(1);
    return v.divide(sum);
  }

  /**
   * Returns n probabilities, one for each category.  If you can use an n-1 coding, and are touchy
   * about allocation performance, then the classify method is probably better to use.
   *
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each category.
   */
  public Vector classifyFull(Vector instance) {
    return classifyFull(new DenseVector(numCategories), instance);
  }

  /**
   * Returns n probabilities, one for each category into a pre-allocated vector.
   *
   * @param r        Where to put the results.
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each category.
   */
  public Vector classifyFull(Vector r, Vector instance) {
    r.viewPart(0, numCategories - 1).assign(classify(instance));
    r.setQuick(numCategories - 1, 1 - r.zSum());
    return r;
  }

  /**
   * Returns a single scalar probability in the case where we have two categories.  Using this
   * method avoids an extra vector allocation as opposed to calling classify() or an extra two
   * vector allocations relative to classifyFull().
   *
   * @param instance The vector of features to be classified.
   * @return The probability of the first of two categories.
   * @throws IllegalArgumentException If the classifier doesn't have two categories.
   */
  public double classifyScalar(Vector instance) {
    if (numCategories() != 2) {
      throw new IllegalArgumentException("Can only call classifyScalar with two categories");
    }

    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    double r = Math.exp(beta.getRow(0).dot(instance));
    return r / (1 + r);
  }

  /**
   * Returns n-1 probabilities, one for each category but the last, for each row of a matrix. The
   * probability of the missing n-th category is 1 - sum(this result).
   *
   * @param data The matrix whose rows are vectors to classify
   * @return A matrix of scores, one row per row of the input matrix, one column for each but the
   *         last category.
   */

  public Matrix classify(Matrix data) {
    Matrix r = new DenseMatrix(data.numRows(), numCategories() - 1);
    for (int row = 0; row < data.numRows(); row++) {
      r.assignRow(row, classify(data.getRow(row)));
    }
    return r;
  }

  /**
   * Returns n probabilities, one for each category, for each row of a matrix. The probability of
   * the missing n-th category is 1 - sum(this result).
   *
   * @param data The matrix whose rows are vectors to classify
   * @return A matrix of scores, one row per row of the input matrix, one column for each but the
   *         last category.
   */
  public Matrix classifyFull(Matrix data) {
    Matrix r = new DenseMatrix(data.numRows(), numCategories());
    for (int row = 0; row < data.numRows(); row++) {
      classifyFull(r.getRow(row).viewPart(0, numCategories() - 1), data.getRow(row));
    }
    return r;
  }

  /**
   * Returns a vector of probabilities of the first category, one for each row of a matrix. This
   * only makes sense if there are exactly two categories, but calling this method in that case can
   * save a number of vector allocations.
   *
   * @param data The matrix whose rows are vectors to classify
   * @return A vector of scores, with one value per row of the input matrix.
   */
  public Vector classifyScalar(Matrix data) {
    if (numCategories() != 2) {
      throw new IllegalArgumentException("Can only call classifyScalar with two categories");
    }

    Vector r = new DenseVector(data.numRows());
    for (int row = 0; row < data.numRows(); row++) {
      r.setQuick(row, classifyScalar(data.getRow(row)));
    }
    return r;
  }

  /**
   * Update the coefficients according to a single instance of known category.
   *
   * @param actual   The category of this instance.
   * @param instance The feature vector for the instance.
   */
  public void train(int actual, Vector instance) {
    sealed = false;

    double learningRate = currentLearningRate();

    // push coefficients back to zero based on the prior
    regularize(instance);

    // what does the current model say?
    Vector v = classify(instance);

    // update each row of coefficients according to result
    for (int i = 0; i < numCategories - 1; i++) {
      double gradientBase = -v.getQuick(i);
      if ((i + 1) == actual) {
        gradientBase += 1;
      }

      // then we apply the gradientBase to the resulting element.
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        Vector.Element updateLocation = nonZeros.next();
        int j = updateLocation.index();
        beta.setQuick(i, j, beta.getQuick(i, j) + learningRate * gradientBase * instance.getQuick(j));
      }
    }

    // TODO can report log likelihood here

    // remember that these elements got updated
    Iterator<Vector.Element> i = instance.iterateNonZero();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      // TODO put confidence weighting here or use per feature annealing
      updateSteps.setQuick(element.index(), step);
    }
    step++;

  }

  private void regularize(Vector instance) {
    if (lambda == 0.0 || prior == null) {
      // no need to regularize
      return;
    }

    // anneal learning rate
    double learningRate = currentLearningRate();

    // here we lazily apply the prior to make up for our neglect
    for (int i = 0; i < numCategories - 1; i++) {
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        Vector.Element updateLocation = nonZeros.next();
        int j = updateLocation.index();
        double missingUpdates = step - updateSteps.getQuick(j);
        if (missingUpdates > 0) {
          // TODO can we put confidence weighting here or use per feature annealing?
          double newValue = prior.age(beta.getQuick(i, j), missingUpdates, lambda * learningRate);
          beta.setQuick(i, j, newValue);
        }
      }
    }
  }

  public double currentLearningRate() {
    return mu_0 * Math.pow(decayFactor, step) * Math.pow(step + stepOffset, forgettingExponent);
  }

  public void setPrior(PriorFunction prior) {
    this.prior = prior;
  }

  public PriorFunction getPrior() {
    return prior;
  }

  public Matrix getBeta() {
    if (!sealed) {
      sealed = true;
      step++;
      regularizeAll();
    }
    return beta;
  }

  private void regularizeAll() {
    Vector all = new DenseVector(beta.numCols());
    all.assign(1);
    regularize(all);
  }

  public void setBeta(int i, int j, double beta_ij) {
    beta.set(i, j, beta_ij);
  }

  public int numCategories() {
    return numCategories;
  }
}
