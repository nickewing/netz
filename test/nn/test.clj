(ns nn.test
  (:use nn.core
        clojure.test
        (incanter core)))

(def negate-network
  {:weights
   [(matrix [[10 -20]])]})

(deftest binary-negate
  (is (= 0 (run-binary negate-network [1])))
  (is (= 1 (run-binary negate-network [0]))))

(def and-network
  {:weights
   [(matrix [[-30 20 20]])]})

(deftest binary-and
  (is (= 0 (run-binary and-network [0 0])))
  (is (= 0 (run-binary and-network [1 0])))
  (is (= 0 (run-binary and-network [0 1])))
  (is (= 1 (run-binary and-network [1 1]))))

(def xnor-network
  {:weights
   [(matrix [[-30  20  20]
             [ 10 -20 -20]])
    (matrix [[-10  20  20]])]})

(deftest binary-xnor
  (is (= 1 (run-binary xnor-network [0 0])))
  (is (= 0 (run-binary xnor-network [1 0])))
  (is (= 0 (run-binary xnor-network [0 1])))
  (is (= 1 (run-binary xnor-network [1 1]))))

(deftest forward-propagation
  (binary-negate)
  (binary-and)
  (binary-xnor))
