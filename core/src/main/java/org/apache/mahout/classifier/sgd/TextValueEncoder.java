package org.apache.mahout.classifier.sgd;

import com.google.common.base.Splitter;
import org.apache.mahout.math.Vector;

import java.util.regex.Pattern;

/**
 * Encodes text that is tokenized on non-alphanum separators.  Each word is encoded using a
 * settable encoder which is by default an StaticWordValueEncoder which gives all
 * words the same weight.
 */
public class TextValueEncoder extends RecordValueEncoder {
  Splitter onNonWord = Splitter.on(Pattern.compile("\\W+")).omitEmptyStrings();
  private RecordValueEncoder wordEncoder;

  public TextValueEncoder(String name) {
    super(name);
    wordEncoder = new StaticWordValueEncoder(name);
    probes = 2;
  }

  /**
   * Adds a value to a vector after tokenizing it by splitting on non-alphanum characters.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, Vector data) {
    for (String word : tokenize(originalForm)) {
      wordEncoder.addToVector(word, data);
    }
  }

  private Iterable<String> tokenize(String originalForm) {
    return onNonWord.split(originalForm);
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the value
   * is being interpreted.  For text-like things, this is likely to be a list of the terms found with
   * associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  @Override
  public String asString(String originalForm) {
    StringBuilder r = new StringBuilder("[");
    String sep = "";
    for (String word : tokenize(originalForm)) {
      r.append(sep);
      r.append(wordEncoder.asString(word));
      sep = ", ";
    }
    r.append("]");
    return r.toString();
  }

  public void setWordEncoder(RecordValueEncoder wordEncoder) {
    this.wordEncoder = wordEncoder;
  }
}
