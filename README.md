Netz
====

Usage

    (ns your-namespace
      (:require [netz.core :as netz]))

    (def inputs [[0 0]
                 [0 1]
                 [1 0]
                 [1 1]])
    (def outputs [[1]
                  [0]
                  [0]
                  [1]])

    (def network (netz/train inputs outputs {:hidden-neurons [2]}))

    (netz/run network [0 0]) ; => [0.9176]
    (netz/run network [0 1]) ; => [0.0549]
    (netz/run network [1 0]) ; => [0.0728]
    (netz/run network [1 1]) ; => [0.9307]
