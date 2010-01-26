package org.apache.mahout.classifier.sgd;

import static java.lang.Math.log;

/**
 * Implements the Gaussian prior. This prior has a tendency to decrease large
 * coefficients toward zero, but doesn't tend to set them to exactly zero.
 *
 * When this prior is used as a regularizer for training a linear model that
 * minimizes the mean squared error loss, the algorithm is called Ridge
 * Regression.
 *
 * When this prior is used as a regularizer for training a linear model that
 * minimizes a hinge loss, the compound algorithm is called linear SVM (a.k.a.
 * Pegasos).
 */
public class L2 extends PriorFunction {
  private final double s2;

  private final double s;

  public L2() {
    this(1.0);
  }

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
