(ns nn.test
  (:use nn.core
        clojure.test
        (incanter core)))

(def xnor-network
  {:weights
   [(matrix [[-30  20  20]
             [ 10 -20 -20]])
    (matrix [[-10  20  20]])]})

(deftest forward-propagation
  (is (= 1 (run-binary xnor-network [0 0])))
  (is (= 0 (run-binary xnor-network [1 0])))
  (is (= 0 (run-binary xnor-network [0 1])))
  (is (= 1 (run-binary xnor-network [1 1]))))
