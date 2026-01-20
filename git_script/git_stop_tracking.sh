# this script is used to stop tracking files or folders
while true; do
	read -p "Enter the name of file/folder you want to stop tracking, or directly enter to exit: " FILE_NAME

	if [ "$FILE_NAME" == "" ]; then
		break
	fi

	# directory
	if [ -d "$FILE_NAME" ]; then
		# Directory - add all files in it
		git rm -r --cached "$FILE_NAME"
        echo -e "\n/$FILE_NAME/" >> .gitignore

	# file
	elif [ -f "$FILE_NAME" ]; then
		git rm --cached "$FILE_NAME"
		echo -e "\n$FILE_NAME" >> .gitignore

	else
		echo "'$FILE_NAME' not found as a file or folder, skipping"
	fi
done

