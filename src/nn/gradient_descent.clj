(in-ns 'nn.core)

(defn- calc-mse
  "Calculate mean squared error of predictions:
  MSE = Σ((actual output - expected output)^2) / no. examples."
  [weights examples]
  (let [sum (reduce
              (fn [total example]
                (let [[input expected-output] example
                      output (forward-propagate weights (matrix input))]
                  (+ total (sum (pow (minus output expected-output) 2)))))
              0
              examples)]
    (/ sum (length examples))))

(defn- gradient-descent-complete?
  [network epoch mse]
  (or (> epoch (option network :max-epochs))
      (< mse (option network :desired-error))))

(defn- call-callback-for-epoch
  [callback epochs-per-callback epoch mse]
  (if (and callback
           (= (mod epoch epochs-per-callback) 0))
    (callback epoch mse false)))

(defn report-callback
  [epoch mse last]
  (if last
    (println "Finished Epoch" epoch "MSE" mse)
    (println "Epoch" epoch "MSE" mse)))

(defn- gradient-descent
  "Preform gradient descent to adjust network weights"
  [network examples]
  (let [callback report-callback ; TODO: allow custom callbacks
        epochs-per-callback 100]
    (loop [weights (:weights network)
           last-changes (new-weight-accumulator (:weights network))
           epoch 0]
      (let [gradients (back-propagate-all-examples network weights examples)
            changes (calc-weight-changes network gradients last-changes)
            new-weights (apply-weight-changes weights changes)
            mse (calc-mse new-weights examples)]
        (call-callback-for-epoch callback epochs-per-callback epoch mse)
        (if (gradient-descent-complete? network epoch mse)
          (do
            (if callback
              (callback epoch mse true))
            (assoc network :weights new-weights))
          (recur new-weights changes (inc epoch)))))))

