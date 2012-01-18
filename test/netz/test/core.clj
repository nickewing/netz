(ns netz.test.core
  (:require [netz.core :as netz])
  (:use clojure.test
        incanter.core))

(def negate-examples-binary
  [[[0] [1]]
   [[1] [0]]])

(def and-examples-binary
  [[[0 0] [0]]
   [[1 0] [0]]
   [[0 1] [0]]
   [[1 1] [1]]])

(def xnor-examples-binary
  [[[0 0] [1]]
   [[1 0] [0]]
   [[0 1] [0]]
   [[1 1] [1]]])

(def or-examples-bipolar
  [[[-1 -1] [0]]
   [[-1  1] [1]]
   [[ 1 -1] [1]]
   [[ 1  1] [1]]])

(def xnor-examples-bipolar
  [[[-1 -1] [1]]
   [[-1  1] [0]]
   [[ 1 -1] [0]]
   [[ 1  1] [1]]])

(def negate-network-binary
  (netz/new-network
   [(matrix [[10 -20]])]))

(def and-network-binary
  (netz/new-network
   [(matrix [[-30 20 20]])]))

(def xnor-network-binary
  (netz/new-network
   [(matrix [[-30  20  20]
             [ 10 -20 -20]])
    (matrix [[-10  20  20]])]))

(defn- test-run-binary
  [network examples]
  (doall (map
           (fn [[input output]]
             (is (= output (netz/run-binary network input))))
           examples)))

(defn- test-train-binary
  [examples options]
  (let [network (netz/train examples options)]
    (test-run-binary network examples)))

(deftest binary-negate-run
  (test-run-binary negate-network-binary negate-examples-binary))

(deftest binary-and-run
  (test-run-binary and-network-binary and-examples-binary))

(deftest binary-xnor-run
  (test-run-binary xnor-network-binary xnor-examples-binary))

(deftest binary-or-train-bprop
  (println "Training OR with bprop")
    (test-train-binary
      or-examples-bipolar
      {:hidden-neurons []
       :training-algorithm :bprop
       :desired-error 0.0005}))

(deftest binary-or-train-rprop
  (println "Training OR with rprop")
    (test-train-binary
      or-examples-bipolar
      {:hidden-neurons []
       :training-algorithm :rprop
       :desired-error 0.0005}))

(deftest binary-xnor-train-bprop
  (println "Training XNOR with bprop")
  (test-train-binary
    xnor-examples-bipolar
    {:hidden-neurons [4]
     :training-algorithm :bprop
     :bprop-learning-rate 0.2
     :bprop-learning-momentum 0.9}))

(deftest binary-xnor-train-rprop
  (println "Training XNOR with rprop")
  (test-train-binary
    xnor-examples-bipolar
    {:hidden-neurons [4]
     :training-algorithm :rprop}))

