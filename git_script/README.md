Currently, git lfs remove remotely hasn't been verified

search for all >100MB files: 

find . -type f -size +100M

create a large file for testing(102.4MB):
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
```

# test for lfs push and remove
## test 1-remove single file
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
```

```bash
fsutil file createnew test_lf_2.txt 107374182 # 102.4 MB
```
## test 2-remove folder
```bash
mkdir file_1 
fsutil file createnew file_1/test_lf_1.txt 107374182 # 102.4 MB
fsutil file createnew file_1/test_lf_2.md 107374182 # 102.4 MB
```

```bash
mkdir file_2
fsutil file createnew file_2/test_lf_3.sh 107374182 # 102.4 MB
fsutil file createnew file_2/test_lf_4.mp4 107374182 # 102.4 MB
```
## test 3-remove with ext
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_2.txt 107374182 # 102.4 MB
mkdir file_1 
fsutil file createnew file_1/test_lf_3.txt 107374182 # 102.4 MB
fsutil file createnew file_1/test_lf_4.txt 107374182 # 102.4 MB
```

```bash
fsutil file createnew test_lf_5.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_6.txt 107374182 # 102.4 MB
mkdir file_2
fsutil file createnew file_2/test_lf_7.txt 107374182 # 102.4 MB
fsutil file createnew file_2/test_lf_8.txt 107374182 # 102.4 MB
```

## test 4-remove all
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_2.md 107374182 # 102.4 MB
mkdir file_1
fsutil file createnew file_1/test_lf_3.jpg 107374182 # 102.4 MB
fsutil file createnew file_1/test_lf_4.mp4 107374182 # 102.4 MB
```

```bash
fsutil file createnew test_lf_5.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_6.md 107374182 # 102.4 MB
mkdir file_2
fsutil file createnew file_2/test_lf_7.jpg 107374182 # 102.4 MB
fsutil file createnew file_2/test_lf_8.mp4 107374182 # 102.4 MB
```
