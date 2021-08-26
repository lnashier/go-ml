package csv

import (
	"bufio"
	"os"
	"path"

	"github.com/go-gota/gota/dataframe"
)

// Save ...
func Save(fpath string, df dataframe.DataFrame) error {
	// Make sure dir exists
	fdir, _ := path.Split(fpath)
	if err := os.MkdirAll(fdir, os.ModePerm); err != nil {
		return err
	}
	f, err := os.Create(fpath)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	if err := df.WriteCSV(w); err != nil {
		return err
	}
	return nil
}
