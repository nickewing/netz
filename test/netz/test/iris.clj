(ns netz.test.iris
  (:require [netz.core :as netz])
  (:use clojure.test
        (incanter core datasets)))

(def iris (to-list (to-matrix (get-dataset :iris))))

(def output-map {0.0 [1 0 0]
                 1.0 [0 1 0]
                 2.0 [0 0 1]})

(defn prepare-example
  [row]
  (let [[input output] (split-at 4 row)
        output (get output-map (first output))]
    [(vec input) output]))

(def iris-examples (map prepare-example iris))

(deftest iris-dataset-test-bprop-rand-init
  (println "Training on iris dataset with bprop and random initialization")
  (time (netz/train iris-examples {:hidden-neurons [3]
                                   :desired-error 0.025
                                   :training-algorithm :bprop
                                   :bprop-learning-rate 0.5
                                   :bprop-learning-momentum 0.5
                                   :weight-initialization-method :random})))

(deftest iris-dataset-test-rprop-rand-init
  (println "Training on iris dataset with rprop and random initialization")
  (time (netz/train iris-examples {:hidden-neurons [3]
                                   :desired-error 0.025
                                   :training-algorithm :rprop
                                   :weight-initialization-method :random})))

(deftest iris-dataset-test-bprop-nguyen-widrow-init
  (println "Training on iris dataset with bprop and Nguyen-Widrow initialization")
  (time (netz/train iris-examples {:hidden-neurons [3]
                                   :desired-error 0.025
                                   :training-algorithm :bprop
                                   :bprop-learning-rate 0.5
                                   :bprop-learning-momentum 0.5
                                   :weight-initialization-method :nguyen-widrow})))

(deftest iris-dataset-test-rprop-nguyen-widrow-init
  (println "Training on iris dataset with rprop and Nguyen-Widrow initialization")
  (time (netz/train iris-examples {:hidden-neurons [3]
                                   :desired-error 0.025
                                   :training-algorithm :rprop
                                   :weight-initialization-method :nguyen-widrow})))
