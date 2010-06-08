package org.apache.mahout.classifier.sgd;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.apache.mahout.math.Vector;

/**
 * Encodes words into vectors much as does WordValueEncoder while maintaining
 * an adaptive dictionary of values seen so far.  This allows weighting of terms
 * without a pre-scan of all of the data.
 */
public class AdaptiveWordValueEncoder extends WordValueEncoder {
  private Multiset<String> dictionary;

  public AdaptiveWordValueEncoder(String name) {
    super(name);
    dictionary = HashMultiset.create();
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, Vector data) {
    dictionary.add(originalForm);
    super.addToVector(originalForm, data);
  }

  @Override
  protected double weight(String originalForm) {
    // the counts here are adjusted so that every observed value has an extra 0.5 count
    // as does a hypothetical unobserved value.  This smooths our estimates a bit and
    // allows the first word seen to have a non-zero weight of -log(1.5 / 2)
    double thisWord = dictionary.count(originalForm) + 0.5;
    double allWords = dictionary.size() + dictionary.elementSet().size() * 0.5 + 0.5;
    return -Math.log(thisWord / allWords);
  }
}
