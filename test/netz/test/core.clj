(ns netz.test.core
  (:use netz.core
        clojure.test
        incanter.core))

(def negate-network
  (new-network
   [(matrix [[10 -20]])]))

(deftest binary-negate-run
  (is (= [0] (run-binary negate-network [1])))
  (is (= [1] (run-binary negate-network [0]))))

(def and-network
  (new-network
   [(matrix [[-30 20 20]])]))

(deftest binary-and-run
  (is (= [0] (run-binary and-network [0 0])))
  (is (= [0] (run-binary and-network [1 0])))
  (is (= [0] (run-binary and-network [0 1])))
  (is (= [1] (run-binary and-network [1 1]))))

(def xnor-network
  (new-network
   [(matrix [[-30  20  20]
             [ 10 -20 -20]])
    (matrix [[-10  20  20]])]))

(deftest binary-xnor-run
  (is (= [1] (run-binary xnor-network [0 0])))
  (is (= [0] (run-binary xnor-network [1 0])))
  (is (= [0] (run-binary xnor-network [0 1])))
  (is (= [1] (run-binary xnor-network [1 1]))))

(deftest binary-or-train
  (let [examples [[[0 0] [0]]
                  [[0 1] [1]]
                  [[1 0] [1]]
                  [[1 1] [1]]]
        network (train examples {:hidden-neurons []})]
    (is (= [0] (run-binary network [0 0])))
    (is (= [1] (run-binary network [1 0])))
    (is (= [1] (run-binary network [0 1])))
    (is (= [1] (run-binary network [1 1])))))

(deftest binary-xnor-train
  (let [examples [[[0 0] [1]]
                  [[0 1] [0]]
                  [[1 0] [0]]
                  [[1 1] [1]]]
        network (train examples {:hidden-neurons [2]
                                :learning-rate 0.2
                                :learning-momentum 0.9})]
    (is (= [1] (run-binary network [0 0])))
    (is (= [0] (run-binary network [1 0])))
    (is (= [0] (run-binary network [0 1])))
    (is (= [1] (run-binary network [1 1])))))
