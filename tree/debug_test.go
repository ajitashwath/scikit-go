package tree

import (
	"fmt"
	"testing"
)

func TestDebugDumpStructure(t *testing.T) {
	fx := loadTreeFixtures(t)
	for _, name := range []string{"cls_xor_gini", "cls_gini", "cls_entropy", "cls_gini_depth2", "reg_mse", "reg_mae", "reg_mse_depth3"} {
		d := fx[name]
		var m *treeImpl
		var crit string
		switch {
		case name == "cls_xor_gini" || name == "cls_gini":
			crit = "gini"
		case name == "cls_entropy":
			crit = "entropy"
		case name == "cls_gini_depth2":
			crit = "gini"
		default:
			crit = "mse"
		}
		if name == "reg_mae" {
			crit = "mae"
		}
		var err error
		if name == "reg_mse" || name == "reg_mae" || name == "reg_mse_depth3" {
			md := NewDecisionTreeRegressor()
			md.Criterion = crit
			if name == "reg_mse_depth3" {
				md.MaxDepth = 3
			}
			if err = md.Fit(d.X, d.Y); err != nil {
				t.Fatalf("%s Fit: %v", name, err)
			}
			m = md.impl
		} else {
			mc := NewDecisionTreeClassifier()
			mc.Criterion = crit
			if name == "cls_gini_depth2" {
				mc.MaxDepth = 2
			}
			if err = mc.Fit(d.X, d.Y); err != nil {
				t.Fatalf("%s Fit: %v", name, err)
			}
			m = mc.impl
		}
		fmt.Printf("== %s ==\n", name)
		for i, n := range m.nodes {
			feat := -1
			thr := ""
			left, right := -1, -1
			if n.Left >= 0 {
				feat = n.Feature
				thr = fmt.Sprintf("%.8f", n.Threshold)
				left, right = n.Left, n.Right
			}
			fmt.Printf("(%d, %s, %.6f, %d, %d, %s, %d, %d)\n",
				i, map[bool]string{true: "split", false: "leaf"}[feat != -1],
				n.Impurity, n.NSamples, feat, thr, left, right)
		}
	}
}
