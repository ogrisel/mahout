package org.apache.mahout.classifier.sgd;

import static java.lang.Math.log;
import static org.apache.commons.math.special.Gamma.logGamma;

/**
 * Provides a t-distribution as a prior.
 *
 * TODO: give expected benefits of using this prior as a regularizer vs Gaussian
 * (L2) or Laplacian (L1).
 */
public class TPrior extends PriorFunction {
  private final double df;

  public TPrior(double df) {
    this.df = df;
  }

  @Override
  public double age(double oldValue, double generations, double learningRate) {
    for (int i = 0; i < generations; i++) {
      oldValue = oldValue - learningRate * oldValue * (df + 1)
          / (df + oldValue * oldValue);
    }
    return oldValue;
  }

  @Override
  public double logP(double beta_ij) {
    return logGamma((df + 1) / 2) - log(df * Math.PI) - logGamma(df / 2)
        - (df + 1) / 2 * log(1 + beta_ij * beta_ij);
  }
}
