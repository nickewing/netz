(ns netz.test.iris
  (:require [netz.core :as netz])
  (:use netz.core
        clojure.test
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

(deftest iris-dataset-test
  (let [examples (map prepare-example iris)]
    (time (netz/train examples {:hidden-neurons [3]
                                :desired-error 0.025
                                :learning-rate 0.5
                                :learning-momentum 0.5}))))
