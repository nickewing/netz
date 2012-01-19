(in-ns 'netz.core)

(defn- random-list
  "Create a list of random doubles between -init-epsilon and +init-epsilon."
  [len init-epsilon]
  (for [x (range 0 len)]
    (- (rand (* 2 init-epsilon)) init-epsilon)))

(defn- random-initial-weights
  "Generate random initial weight matrices for given layer-sizes."
  [layer-sizes init-epsilon]
  (for [i (range 0 (dec (length layer-sizes)))]
    (let [cols (inc (get layer-sizes i))
          rows (get layer-sizes (inc i))]
      (matrix (random-list (* rows cols) init-epsilon) cols))))

(defn- nguyen-widrow-initial-weights
  "Generate initial weights based on Nguyen-Widrow algorithm."
  [layer-sizes init-epsilon]
  (let [num-hidden (sum (subvec layer-sizes 1 (dec (length layer-sizes))))
        weights (random-initial-weights layer-sizes init-epsilon)]
    (if (< num-hidden 1)
      weights
      (let [num-inputs (first layer-sizes)
            beta (* 0.7 (pow num-hidden (/ 1.0 num-inputs)))]
        (map
          (fn [weight]
            (let [[rows cols] (dim weight)]
              (matrix
                (for [i (range rows)]
                  (let [row (sel weight :rows i)
                        n (sqrt (sum (pow row 2)))]
                    (div (mult beta row) n))))))
          weights)))))
