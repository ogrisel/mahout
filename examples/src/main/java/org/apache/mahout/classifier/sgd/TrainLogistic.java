package org.apache.mahout.classifier.sgd;

import com.google.common.collect.Maps;
import com.google.common.io.LineReader;
import com.google.common.io.Resources;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.Iterator;
import java.util.Map;


/**
 * Train a logistic regression for the examples from Chapter 13 of Mahout in Action
 */
public class TrainLogistic {
  private static final Logger log = LoggerFactory.getLogger(TrainLogistic.class);
  private static String inputFile;
  private static String outputFile;
  private static String targetVariable;
  private static int targetCategories;
  private static Map<String, String> typeMap;
  private static int numFeatures;
  private static double lambda;
  private static double learningRate;
  private static int passes;
  private static boolean useBias;


  public static void main(String[] args) throws IOException {
    if (parseArgs(args)) {
      double logPEstimate = 0;
      int samples = 0;

      OnlineLogisticRegression lr = new OnlineLogisticRegression(targetCategories, numFeatures, new L1())
              .lambda(lambda)
              .learningRate(learningRate)
              .alpha(1 - 1e-3);

      CsvRecordFactory csv = new CsvRecordFactory(targetCategories)
              .includeBiasTerm(useBias);
      for (int pass = 0; pass < passes; pass++) {
        InputStreamReader s;
        try {
          URL resource = Resources.getResource(inputFile);
          s = new InputStreamReader(resource.openStream());
        } catch (IllegalArgumentException e) {
          s = new FileReader(inputFile);
        }
        LineReader in = new LineReader(s);

        // read variable names
        csv.firstLine(targetVariable, typeMap, in.readLine());

        String line = in.readLine();
        while (line != null) {
          // for each new line, get target and predictors
          Vector input = new RandomAccessSparseVector(numFeatures);
          int targetValue = csv.processLine(line, input);

          // check performance while this is still news
          double logP = logLikelihood(lr, input, targetValue);
          if (!Double.isInfinite(logP)) {
            if (samples < 20) {
              logPEstimate = (samples * logPEstimate + logP) / (samples + 1);
            } else {
              logPEstimate = 0.95 * logPEstimate + 0.05 * logP;
            }
            samples++;
          }
          double p = lr.classifyScalar(input);
          System.out.printf("%10d %2d %10.2f %2.4f %10.4f %10.4f\n", samples, targetValue, lr.currentLearningRate(), p, logP, logPEstimate);

          // now update model
          lr.train(targetValue, input);

          line = in.readLine();
        }

      }
      System.out.printf("%d\n", numFeatures);
      System.out.printf("%s ~ ", targetVariable);
      String sep = "";
      for (String v : csv.getPredictors()) {
        double weight = predictorWeight(lr, 0, csv, v);
        if (weight != 0) {
          System.out.printf("%s%.3f*%s", sep, weight, v);
          sep = " + ";
        }
      }
      System.out.printf("\n");
      for (int row = 0; row < lr.getBeta().numRows(); row++) {
        for (String key : csv.getTraceDictionary().keySet()) {
          double weight = predictorWeight(lr, row, csv, key);
          if (weight != 0) {
            System.out.printf("%20s %.5f\n", key, weight);
          }
        }
        for (int column = 0; column < lr.getBeta().numCols(); column++) {
          System.out.printf("%15.9f ", lr.getBeta().get(row, column));
        }
        System.out.println();
      }
    }
  }

  private static double predictorWeight(OnlineLogisticRegression lr, int row, CsvRecordFactory csv, String predictor) {
    double weight = 0;
    for (Integer column : csv.getTraceDictionary().get(predictor)) {
      weight += lr.getBeta().get(row, column);
    }
    return weight;
  }

  private static double logLikelihood(OnlineLogisticRegression model, Vector data, int category) {
    if (model.numCategories() == 2) {
      double p = model.classifyScalar(data);
      if (category > 0) {
        return Math.log(p);
      } else {
        return Math.log(1 - p);
      }
    } else {
      Vector p = model.classify(data);
      if (category < model.numCategories()) {
        return Math.log(p.get(category));
      } else {
        return Math.log(1 - p.zSum());
      }
    }
  }

  private static boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    Option quiet = builder.withLongName("quiet").withDescription("be extra quiet").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFile = builder.withLongName("input")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
            .withDescription("where to get training data")
            .create();

    Option outputFile = builder.withLongName("output")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
            .withDescription("where to get training data")
            .create();

    Option predictors = builder.withLongName("predictors")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("p").create())
            .withDescription("a list of predictor variables")
            .create();

    Option types = builder.withLongName("types")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("t").create())
            .withDescription("a list of predictor variable types (numeric, word, or text)")
            .create();

    Option target = builder.withLongName("target")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("target").withMaximum(1).create())
            .withDescription("the name of the target variable")
            .create();

    Option features = builder.withLongName("features")
            .withArgument(
                    argumentBuilder.withName("numFeatures")
                            .withDefault("1000")
                            .withMaximum(1).create())
            .withDescription("the number of internal hashed features to use")
            .create();

    Option passes = builder.withLongName("passes")
            .withArgument(
                    argumentBuilder.withName("passes")
                            .withDefault("2")
                            .withMaximum(1).create())
            .withDescription("the number of times to pass over the input data")
            .create();

    Option lambda = builder.withLongName("lambda")
            .withArgument(argumentBuilder.withName("lambda").withDefault("1e-4").withMaximum(1).create())
            .withDescription("the amount of coefficient decay to use")
            .create();

    Option rate = builder.withLongName("rate")
            .withArgument(argumentBuilder.withName("learningRate").withDefault("1e-3").withMaximum(1).create())
            .withDescription("the learning rate")
            .create();

    Option noBias = builder.withLongName("noBias")
            .withDescription("don't include a bias term")
            .create();

    Option targetCategories = builder.withLongName("categories")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("number").withMaximum(1).create())
            .withDescription("the number of target categories to be considered")
            .create();

    Group normalArgs = new GroupBuilder()
            .withOption(help)
            .withOption(quiet)
            .withOption(inputFile)
            .withOption(outputFile)
            .withOption(target)
            .withOption(targetCategories)
            .withOption(predictors)
            .withOption(types)
            .withOption(passes)
            .withOption(lambda)
            .withOption(rate)
            .withOption(noBias)
            .withOption(features)
            .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine;
    cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    TrainLogistic.inputFile = getStringArgument(cmdLine, inputFile);
    TrainLogistic.outputFile = getStringArgument(cmdLine, inputFile);

    TrainLogistic.typeMap = Maps.newHashMap();
    if (cmdLine.getValues(types).size() == 0) {
      throw new IllegalArgumentException("Must have at least one type specifier");
    }
    Iterator iTypes = cmdLine.getValues(types).iterator();
    String lastType = null;
    for (Object x : cmdLine.getValues(predictors)) {
      // type list can be short .. we just repeat last spec
      if (iTypes.hasNext()) {
        lastType = iTypes.next().toString();
      }
      typeMap.put(x.toString(), lastType);
    }

    TrainLogistic.targetVariable = getStringArgument(cmdLine, target);
    TrainLogistic.targetCategories = getIntegerArgument(cmdLine, targetCategories);

    TrainLogistic.numFeatures = getIntegerArgument(cmdLine, features);
    TrainLogistic.passes = getIntegerArgument(cmdLine, passes);
    TrainLogistic.lambda = getDoubleArgument(cmdLine, lambda);
    TrainLogistic.learningRate = getDoubleArgument(cmdLine, rate);

    TrainLogistic.useBias = !getBooleanArgument(cmdLine, noBias);
    return true;
  }

  private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
    return (String) cmdLine.getValue(inputFile);
  }

  private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
    return cmdLine.hasOption(option);
  }

  private static int getIntegerArgument(CommandLine cmdLine, Option features) {
    return Integer.parseInt((String) cmdLine.getValue(features));
  }

  private static double getDoubleArgument(CommandLine cmdLine, Option op) {
    return Double.parseDouble((String) cmdLine.getValue(op));
  }


}
