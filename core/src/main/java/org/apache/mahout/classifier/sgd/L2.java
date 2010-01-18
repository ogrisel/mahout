package org.apache.mahout.classifier.sgd;

import static java.lang.Math.log;

/**
 * Implements the Gaussian prior.  This prior has a tendency to decrease large coefficients toward zero, but
 * doesn't tend to set them to exactly zero.
 */
public class L2 extends PriorFunction {
  private double s2;
  private double s;

  public L2(double scale) {
    this.s = scale;
    this.s2 = scale * scale;
  }

  @Override
  public double age(double oldValue, double generations, double learningRate) {
    return oldValue * Math.pow(1 - learningRate / s2, generations);
  }

  @Override
  public double logP(double beta_ij) {
    return -beta_ij * beta_ij / s2 / 2 - log(s) - log(2 * Math.PI) / 2;
  }
}
