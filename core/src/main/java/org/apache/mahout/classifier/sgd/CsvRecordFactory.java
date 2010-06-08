package org.apache.mahout.classifier.sgd;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.mahout.math.Vector;

import java.lang.reflect.InvocationTargetException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Converts csv data lines to vectors.
 */
public class CsvRecordFactory {
  // crude CSV value splitter.  This will fail if any double quoted strings have
  // commas inside.  Also, escaped quotes will not be unescaped.  Good enough for now.
  private Splitter onComma = Splitter.on(",").trimResults(CharMatcher.is('"'));

  private static final Map<String, Class<? extends RecordValueEncoder>> typeDictionary =
          ImmutableMap.<String, Class<? extends RecordValueEncoder>>builder()
                  .put("continuous", ContinuousValueEncoder.class)
                  .put("numeric", ContinuousValueEncoder.class)
                  .put("n", ContinuousValueEncoder.class)
                  .put("word", StaticWordValueEncoder.class)
                  .put("w", StaticWordValueEncoder.class)
                  .put("text", TextValueEncoder.class)
                  .put("t", TextValueEncoder.class)
                  .build();

  private Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();

  private int target;
  private Dictionary targetDictionary;

  private List<Integer> predictors;
  private Map<Integer, RecordValueEncoder> predictorEncoders;
  private int targetCategories;
  private List<String> variableNames;
  private boolean includeBiasTerm;
  private static final String INTERCEPT_TERM = "Intercept Term";


  public CsvRecordFactory(int targetCategories) {
    this.targetCategories = targetCategories;
  }

  /**
   * Processes the first line of a file (which should contain the variable names). The target and
   * predictor column numbers are set from the names on this line.
   *
   * @param targetName The name of the target variable.
   * @param typeMap    A map from predictor variable names to encoding name (number, word or text)
   * @param line       Header line for the file.
   */
  public void firstLine(String targetName, final Map<String, String> typeMap, String line) {
    // read variable names, build map of name -> column
    final Map<String, Integer> vars = Maps.newHashMap();
    int column = 0;
    variableNames = Lists.newArrayList(onComma.split(line));
    for (String var : variableNames) {
      vars.put(var, column++);
    }

    // record target column and establish dictionary for decoding target
    target = vars.get(targetName);
    targetDictionary = new Dictionary();

    // create list of predictor column numbers
    predictors = Lists.newArrayList(Collections2.transform(typeMap.keySet(), new Function<String, Integer>() {
      @Override
      public Integer apply(String from) {
        Integer r = vars.get(from);
        if (r == null) {
          throw new IllegalArgumentException("Can't find variable " + from + ", only know about " + vars);
        }
        return r;
      }
    }));

    if (includeBiasTerm) {
      predictors.add(-1);
    }
    Collections.sort(predictors);

    // and map from column number to type encoder for each column that is a predictor
    predictorEncoders = Maps.newHashMap();
    for (Integer predictor : predictors) {
      String name;
      Class<? extends RecordValueEncoder> c;
      if (predictor != -1) {
        name = variableNames.get(predictor);
        c = typeDictionary.get(typeMap.get(name));
      } else {
        name = INTERCEPT_TERM;
        c = ConstantValueEncoder.class;
      }
      try {
        RecordValueEncoder encoder = c.getConstructor(String.class).newInstance(name);
        predictorEncoders.put(predictor, encoder);
        encoder.setTraceDictionary(traceDictionary);
      } catch (InstantiationException e) {
        throw new ImpossibleException("Unable to construct type converter... shouldn't be possible", e);
      } catch (IllegalAccessException e) {
        throw new ImpossibleException("Unable to construct type converter... shouldn't be possible", e);
      } catch (InvocationTargetException e) {
        throw new ImpossibleException("Unable to construct type converter... shouldn't be possible", e);
      } catch (NoSuchMethodException e) {
        throw new ImpossibleException("Unable to construct type converter... shouldn't be possible", e);
      }
    }
  }

  /**
   * Returns a list of the names of the predictor variables.
   *
   * @return A list of variable names.
   */
  public Iterable<String> getPredictors() {
    return Lists.transform(predictors, new Function<Integer, String>() {
      @Override
      public String apply(Integer v) {
        if (v >= 0) {
          return variableNames.get(v);
        } else {
          return INTERCEPT_TERM;
        }
      }
    });
  }

  public Map<String, Set<Integer>> getTraceDictionary() {
    return traceDictionary;
  }

  public CsvRecordFactory includeBiasTerm(boolean useBias) {
    includeBiasTerm = useBias;
    return this;
  }

  private static class ImpossibleException extends RuntimeException {
    private ImpossibleException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /**
   * Decodes a single line of csv data and records the target and predictor variables in a record.
   * As a side effect, features are added into the featureVector.
   *
   * @param line          The raw data.
   * @param featureVector Where to fill in the features.  Should be zeroed before calling
   *                      processLine.
   * @return The value of the target variable.
   */
  public int processLine(String line, Vector featureVector) {
    List<String> values = Lists.newArrayList(onComma.split(line));

    int targetValue = targetDictionary.intern(values.get(target));
    if (targetValue >= targetCategories) {
      targetValue = targetCategories - 1;
    }

    for (Integer predictor : predictors) {
      String value;
      if (predictor >= 0) {
        value = values.get(predictor);
      } else {
        value = null;
      }
      predictorEncoders.get(predictor).addToVector(value, featureVector);
    }
    return targetValue;
  }

  private static class Dictionary {
    private Map<String, Integer> dict = Maps.newHashMap();

    public int intern(String s) {
      if (!dict.containsKey(s)) {
        dict.put(s, dict.size());
      }
      return dict.get(s);
    }
  }
}
