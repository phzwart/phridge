;;; Coot Scheme script to load 1IEE model and all maps
;;; Run via: coot --script examples/lysozyme/load_coot.scm

(let* ((pdb-file "examples/lysozyme/1iee_out_model.pdb")
       (map-2fofc "examples/lysozyme/1iee_out_2fofc.ccp4")
       (map-fofc "examples/lysozyme/1iee_out_fofc.ccp4")
       (map-grad "examples/lysozyme/1iee_out_gradient.ccp4")
       (imol-pdb (handle-read-draw-molecule pdb-file))
       (imol-2f (handle-read-ccp4-map map-2fofc 0))
       (imol-fo (handle-read-ccp4-map map-fofc 1))
       (imol-gr (handle-read-ccp4-map map-grad 1)))
  (set-contour-level-in-sigma imol-2f 1.5)
  (set-contour-level-in-sigma imol-fo 3.0)
  (set-contour-level-in-sigma imol-gr 3.0)
  (display "All maps and model loaded into Coot.\n"))
