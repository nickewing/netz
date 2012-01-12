(ns nn.test
  (:use nn.core
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
  (let [inputs [[0 0] [0 1] [1 0] [1 1]]
        outputs [[0] [1] [1] [1]]
        network (train inputs outputs {:hidden-neurons []})]
    (is (= [0] (run-binary network [0 0])))
    (is (= [1] (run-binary network [1 0])))
    (is (= [1] (run-binary network [0 1])))
    (is (= [1] (run-binary network [1 1])))))

(deftest binary-xnor-train
  (let [inputs [[0 0] [0 1] [1 0] [1 1]]
        outputs [[1] [0] [0] [1]]
        network (train inputs outputs {:hidden-neurons [2]
                                       :learning-rate 0.2
                                       :learning-momentum 0.9})]
    (is (= [1] (run-binary network [0 0])))
    (is (= [0] (run-binary network [1 0])))
    (is (= [0] (run-binary network [0 1])))
    (is (= [1] (run-binary network [1 1])))))

; (deftest binary-xnor-xor-train
  ; (let [inputs [[0 0 0]
                ; [0 1 0]
                ; [1 0 0]
                ; [1 1 0]
                ; [0 0 1]
                ; [0 1 1]
                ; [1 0 1]
                ; [1 1 1]]
        ; outputs [[1 0]
                 ; [0 0]
                 ; [0 1]
                 ; [1 1]
                 ; [1 1]
                 ; [0 1]
                 ; [0 0]
                 ; [1 0]]
        ; network (train inputs outputs {:hidden-neurons [3]
                                       ; :learning-rate 0.2
                                       ; :learning-momentum 0.9})]
    ; (is (= [1 0] (run-binary network [0 0 0])))
    ; (is (= [0 0] (run-binary network [0 1 0])))
    ; (is (= [0 1] (run-binary network [1 0 0])))
    ; (is (= [1 1] (run-binary network [1 1 0])))
    ; (is (= [1 1] (run-binary network [0 0 1])))
    ; (is (= [0 1] (run-binary network [0 1 1])))
    ; (is (= [0 0] (run-binary network [1 0 1])))
    ; (is (= [1 0] (run-binary network [1 1 1])))))
