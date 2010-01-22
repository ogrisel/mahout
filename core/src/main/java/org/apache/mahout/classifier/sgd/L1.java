package org.apache.mahout.classifier.sgd;

/**
 * Implements the Laplacian or bi-exponential prior. This prior has a string
 * tendency to set coefficients to zero and thus is useful as an alternative to
 * variable selection. This version implements truncation which prevents a
 * coefficient from changing sign. If a correction would change the sign, the
 * coefficient is truncated to zero, see:
 *
 * http://jmlr.csail.mit.edu/papers/volume10/langford09a/langford09a.pdf
 * http://jmlr.csail.mit.edu/papers/volume10/duchi09a/duchi09a.pdf
 *
 * When this prior is used as a regularizer for training a linear model that
 * minimizes the mean squared error loss, the algorithm is called the LASSO.
 *
 * Note that it doesn't matter to have a scale for this distribution because
 * after taking the derivative of the logP, the lambda coefficient used to
 * combine the prior with the observations has the same effect. If we had a
 * scale here, then it would be the same effect as just changing lambda.
 */
public class L1 extends PriorFunction {
  @Override
  public double age(double oldValue, double generations, double learningRate) {
    double newValue = oldValue - Math.signum(oldValue) * learningRate
        * generations;
    if (newValue * oldValue < 0) {
      // don't allow the value to change sign
      return 0;
    } else {
      return newValue;
    }
  }

  @Override
  public double logP(double beta_ij) {
    return -Math.abs(beta_ij);
  }

  @Override
  public boolean isSparsityInducing() {
    return true;
  }
}
