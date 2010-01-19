package org.apache.mahout.classifier.sgd;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Generic definition of a 1 of n logistic regression classifier that returns probabilities in response to a feature vector.  This
 * classifier uses 1 of n-1 coding.
 * <p/>
 * TODO: implement Gaussian, Laplacian and t priors
 * TODO: implement per coefficient annealing schedule
 * TODO: implement symbolic input with string, overall cooccurrence and n-gram hash encoding
 * TODO: implement reporter system to monitor progress
 */
public class OnlineLogisticRegression {
  // coefficients for the classification
  private final Matrix beta;

  // number of categories we are classifying.  This should the number of rows of beta plus one.
  private final int numCategories;

  // information about how long since coefficient rows were updated
  private int step = 0;
  private Vector updateSteps;

  // learning rate and decay factor
  private double mu_0 = 1;
  private double alpha = 1 - 1e-3;

  // prior and weight
  private double lambda = 0.1;
  private PriorFunction prior;

  // conversion from term lists to vectors
  private TermRandomizer randomizer;

  public OnlineLogisticRegression(int numCategories, int numFeatures, PriorFunction prior) {
    this(numCategories, numFeatures, prior, RandomUtils.getRandom());
  }

  public OnlineLogisticRegression(int numCategories, int numFeatures, PriorFunction prior, Random rng) {
    this.numCategories = numCategories;
    this.prior = prior;

    updateSteps = new DenseVector(numFeatures);
    beta = new DenseMatrix(numCategories - 1, numFeatures);
    for (int row = 0; row < numCategories - 1; row++) {
      for (int column = 0; column < numFeatures; column++) {
        beta.set(row, column, rng.nextGaussian() * 0.001);
      }
    }
  }

  /**
   * Chainable configuration option.
   * @param alpha New value of alpha, the exponential decay rate for the learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  /**
   * Chainable configuration option.
   * @param learningRate New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression learningRate(double learningRate) {
    this.mu_0 = learningRate;
    return this;
  }

  /**
   * Chainable configuration option.
   * @param lambda New value of lambda, the weighting factor for the prior distribution.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression lambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  /**
   * Returns n-1 probabilities, one for each category but the first.  The probability of the first
   * category is 1 - sum(this result).  The input is in the form of a list of Strings which is
   * interpreted as a sequence of terms.  Each term is hashed and folded down to some relatively
   * small number of features to be given to the underlying classifier.  IN addition, optionally
   * all pairs of strings or all pairs of strings that occur within a specified window can be
   * used as features as well.
   *
   * @param terms    The list of terms to use as input vector.
   * @param window   If > 0, then pairs of terms from a sliding window will be added in.
   * @param allPairs If true, then all pairs of terms in the input list will be added in.
   * @return A vector of scores for the different categories using 1 of n-1 coding.
   */
  public Vector classify(List<String> terms, int window, boolean allPairs) {
    if (randomizer == null) {
      throw new IllegalArgumentException("Term randomizer must be set using setRandomizer before classifying term list");
    }
    return classify(randomizer.randomizedInstance(terms, window, allPairs));
  }

  /**
   * Returns n-1 probabilities, one for each category but the first.  The probability of the first
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
   * Returns n probabilities, one for each category.  If you can use an n-1 coding, and are
   * touchy about allocation performance, then the classify method is probably better to use.
   *
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each category.
   */
  public Vector classifyFullVector(Vector instance) {
    Vector v = classify(instance);
    Vector r = new DenseVector(numCategories);
    r.setQuick(0, 1 - v.zSum());
    r.viewPart(1, numCategories - 1).assign(v);
    return r;
  }

  /**
   * Update the coefficients according to a single instance of known category.
   *
   * @param actual   The category of this instance.
   * @param instance The feature vector for the instance.
   */
  public void train(int actual, Vector instance) {
    step++;

    double learningRate = currentLearningRate();

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
        beta.set(i, j, beta.get(i, j) + learningRate * gradientBase * instance.get(j));
      }
    }

    if (prior.isSparsityInducing()) {
      // eagerly re-apply regularization for the last update step to benefit
      // from promised sparsity in the parameters
      regularize(instance, true);
    }

    // TODO can report log likelihood here
  }

  private void regularize(Vector instance) {
    regularize(instance, false);
  }

  private void regularize(Vector instance, boolean forceOne) {
    // anneal learning rate
    double learningRate = currentLearningRate();

    // here we lazily apply the prior to make up for our neglect
    for (int i = 0; i < numCategories - 1; i++) {
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        int j = nonZeros.next().index();
        double missingUpdates = forceOne ? 1 : step - updateSteps.get(j);
        if (missingUpdates > 0) {
          // TODO can we put confidence weighting here or use per feature annealing?
          beta.set(i, j, prior.age(beta.get(i, j), missingUpdates, lambda * learningRate));
        }
      }
    }
    if (!forceOne) {
      // remember that these elements got updated
      Iterator<Vector.Element> i = instance.iterateNonZero();
      while (i.hasNext()) {
        Vector.Element element = i.next();
        // TODO put confidence weighting here or use per feature annealing
        updateSteps.setQuick(element.index(), step);
      }
    }
  }

  private double currentLearningRate() {
    return mu_0 * Math.pow(alpha, step);
  }

  public void setPrior(PriorFunction prior) {
    this.prior = prior;
  }

  public PriorFunction getPrior() {
    return prior;
  }

  public Matrix getBeta() {
    return beta;
  }

  public void setBeta(int i, int j, double beta_ij) {
    beta.set(i, j, beta_ij);
  }

  public void setRandomizer(TermRandomizer randomizer) {
    this.randomizer = randomizer;
  }
}
