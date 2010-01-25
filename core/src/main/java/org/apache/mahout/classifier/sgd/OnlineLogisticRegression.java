package org.apache.mahout.classifier.sgd;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Generic definition of a 1 of n logistic regression classifier that returns
 * probabilities in response to a feature vector. This classifier uses 1 of n-1
 * coding.
 *
 * Regularization is achieved thanks to the the following prior implementations:
 * Uniform (no regularization), L1 - Laplacian, L2 - Gaussian and Student-T.
 *
 * The use of randomizers allow for building features for symbolic input with
 * string, overall cooccurrence and n-gram hash encoding
 *
 * <p/>
 * TODO: implement per coefficient annealing schedule
 * <p/>
 * TODO: implement reporter system to monitor progress: see 'Progressive
 * Validation' by John Langford et al.
 */
public class OnlineLogisticRegression {
  // coefficients for the classification
  private final Matrix beta;

  // number of categories we are classifying. This should the number of rows of
  // beta plus one.
  private final int numCategories;

  // information about how long since coefficient rows were updated
  private int step = 0;
  private final Vector updateSteps;

  // learning rate and decay factor
  private double mu_0 = 1;
  private double alpha = 1 - 1e-3;

  // prior and weight
  private double lambda = 0.1;
  private PriorFunction prior;

  // conversion from term lists to vectors
  private TermRandomizer randomizer;

  public OnlineLogisticRegression(int numCategories, int numFeatures,
      PriorFunction prior) {
    this(numCategories, numFeatures, prior, RandomUtils.getRandom());
  }

  public OnlineLogisticRegression(int numCategories, int numFeatures,
      PriorFunction prior, Random rng) {
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
   *
   * @param alpha
   *          New value of alpha, the exponential decay rate for the learning
   *          rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param learningRate
   *          New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression learningRate(double learningRate) {
    this.mu_0 = learningRate;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param lambda
   *          New value of lambda, the weighting factor for the prior
   *          distribution.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression lambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  /**
   * Returns n-1 probabilities, one for each category but the first. The
   * probability of the first category is 1 - sum(this result). The input is in
   * the form of a list of Strings which is interpreted as a sequence of terms.
   * Each term is hashed and folded down to some relatively small number of
   * features to be given to the underlying classifier. IN addition, optionally
   * all pairs of strings or all pairs of strings that occur within a specified
   * window can be used as features as well.
   *
   * @param terms
   *          The list of terms to use as input vector.
   * @param window
   *          If > 0, then pairs of terms from a sliding window will be added
   *          in.
   * @param allPairs
   *          If true, then all pairs of terms in the input list will be added
   *          in.
   * @return A vector of scores for the different categories using 1 of n-1
   *         coding.
   */
  public Vector classify(List<String> terms, int window, boolean allPairs) {
    if (randomizer == null) {
      throw new IllegalArgumentException(
          "Term randomizer must be set using setRandomizer before classifying term list");
    }
    return classify(randomizer.randomizedInstance(terms, window, allPairs));
  }

  /**
   * Returns n-1 probabilities, one for each category but the first. The
   * probability of the first category is 1 - sum(this result).
   *
   * @param instance
   *          A vector of features to be classified.
   * @return A vector of probabilities, one for each of the first n-1
   *         categories.
   */
  public Vector classify(Vector instance) {
    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    Vector v = beta.times(instance).assign(Functions.exp);
    double sum = 1 + v.norm(1);
    return v.divide(sum);
  }

  /**
   * Returns n probabilities, one for each category. If you can use an n-1
   * coding, and are touchy about allocation performance, then the classify
   * method is probably better to use.
   *
   * @param instance
   *          A vector of features to be classified.
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
   * @param actual
   *          The category of this instance.
   * @param instance
   *          The feature vector for the instance. For best performances vector
   *          implementation should feature fast sequential access of non zero
   *          components such as SequentialAccessSparseVector
   */
  public void train(int actual, Vector instance) {
    train(actual, instance, null);
  }

  /**
   * Update the coefficients according to a single instance of known category.
   *
   * @param actual
   *          The category of this instance.
   * @param instance
   *          The feature vector for the instance.
   * @param probabilities
   *          Pre-computed probabilities (optional)
   */
  public void train(int actual, Vector instance, Vector probabilities) {
    // what does the current model say?
    if (probabilities == null) {
      probabilities = classify(instance);
    }

    double learningRate = currentLearningRate();

    // update each row of coefficients according to result
    for (int i = 0; i < numCategories - 1; i++) {
      double gradientBase = -probabilities.getQuick(i);
      if ((i + 1) == actual) {
        gradientBase += 1;
      }

      // then we apply the gradientBase to the resulting element.
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        Vector.Element element = nonZeros.next();
        int j = element.index();
        beta.setQuick(i, j, beta.getQuick(i, j) + learningRate * gradientBase
            * element.get());
      }
    }

    // TODO can report log likelihood here

    // increment step after the update to ensure that lazy regularization
    // that happens at classification plays well, e.s.p. for sparsity inducing
    // priors
    step++;
  }

  private void regularize(Vector instance) {
    // anneal learning rate
    double learningRate = currentLearningRate();

    // here we lazily apply the prior to make up for our neglect
    for (int i = 0; i < numCategories - 1; i++) {
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        int j = nonZeros.next().index();
        double missingUpdates = step - updateSteps.getQuick(j);
        if (missingUpdates > 0) {
          // TODO can we put confidence weighting here or use per feature
          // annealing?
          beta.setQuick(i, j, prior.age(beta.getQuick(i, j), missingUpdates,
              lambda * learningRate));
        }
      }
    }
    // remember that these elements got updated
    Iterator<Vector.Element> i = instance.iterateNonZero();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      // TODO put confidence weighting here or use per feature annealing
      updateSteps.setQuick(element.index(), step);
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

  public TermRandomizer getRandomizer() {
    return randomizer;
  }
}
