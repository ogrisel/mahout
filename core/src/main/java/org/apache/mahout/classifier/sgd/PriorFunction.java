package org.apache.mahout.classifier.sgd;

/**
 * A prior is used to regularize the learning algorithm.
 */
public abstract class PriorFunction {
  /**
   * Applies the regularization to a coefficient.
   * @param oldValue        The previous value.
   * @param generations     The number of generations.
   * @param learningRate    The learning rate with lambda baked in.
   * @return                The new coefficient value after regularization.
   */
  public abstract double age(double oldValue, double generations, double learningRate);

  /**
   * Returns the log of the probability of a particular coefficient value according to the prior.
   * @param beta_ij         The coefficient.
   * @return                The log probability.
   */
  public abstract double logP(double beta_ij);
}
