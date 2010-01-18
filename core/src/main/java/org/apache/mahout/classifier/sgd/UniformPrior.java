package org.apache.mahout.classifier.sgd;

/**
 * A uniform prior.  This is an improper prior that corresponds to no regularization at all.
 */
public class UniformPrior extends PriorFunction {
  @Override
  public double age(double oldValue, double generations, double learningRate) {
    return oldValue;
  }

  @Override
  public double logP(double beta_ij) {
    return 0;
  }
}
