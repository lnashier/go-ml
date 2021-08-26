package file

import (
	"io/ioutil"
	"os"
	"path"
)

// Save ...
func Save(fpath string, d []byte) error {
	// Make sure dir exists
	fdir, _ := path.Split(fpath)
	if err := os.MkdirAll(fdir, os.ModePerm); err != nil {
		return err
	}
	if err := ioutil.WriteFile(fpath, d, 0644); err != nil {
		return err
	}
	return nil
}
