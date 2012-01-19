(defn- matrix-mult
  "Multiply two matrices and ensure the result is also a matrix."
  [a b]
  (let [result (mmult a b)]
    (if (matrix? result)
      result
      (matrix [result]))))

(defn- round-output
  "Round outputs to nearest integer."
  [output]
  (vec (map #(Math/round ^Double %) output)))

(defn- sign
  "Returns +1 if x is positive, -1 if negative, 0 otherwise"
  [x]
  (cond (> x 0.0)  1.0
        (< x 0.0) -1.0
        (= x 0.0)  0.0))

(defn- map-to-vectors
  "Map given inputs to func and conj each output to the end of accumulating
  vectors."
  [func & inputs]
  (loop [inputs inputs
         accum nil]
    (let [input (map first inputs)
          inputs (map rest inputs)
          output (apply func input)
          accum (if accum
                  (map conj accum output)
                  (map vector output))]
      (if (empty? (first inputs))
        accum
        (recur inputs accum)))))

