# use git lfs to push large files or folders to remote repository, for specified branch
read -p "Enter the name of your feature branch: " BRANCH_NAME

# this script only pushes those large files, but doesn't push other normal files.

# Uncomment if you need to convert DOS to Unix line endings
# sed -i 's/\r$//' git_script/git_lfs_push.sh

touch "git_lfs_tracked_files.txt"
# Function to scan files and automatically track those over 100MB
scan_and_push_large_files() {
	echo "Scanning all files to identify those over 100MB..."

	# Find all files over 100MB and add them to git-lfs tracking
	find . -type f -size +100M | while read -r file; do
		# Remove the leading './' from the file path if present
		file="${file#./}"
		echo "Tracking large file: $file"
		git lfs track "$file"
		git add "$file"
		if [ ! -s "git_lfs_tracked_files.txt" ]; then
			echo "$file" >>"git_lfs_tracked_files.txt"
		else
			echo -e "\n$file" >>"git_lfs_tracked_files.txt"
		fi
	done

	# remove if empty
	if [ ! -s "git_lfs_tracked_files.txt" ]; then
		rm -f "git_lfs_tracked_files.txt"
	else
		git add "git_lfs_tracked_files.txt"
	fi

	# Stage the .gitattributes file where git-lfs tracks the large files
	git add .gitattributes

	# Commit and push
	git commit -m "update"
	git push origin "$BRANCH_NAME"
}

# Initialize Git LFS
git lfs install

# Ask the user if they want to scan all files and auto-track those over 100MB
read -p "Do you want to scan all files, track those over 100MB, and push them automatically? (yes/no): " SCAN_CHOICE

# Process based on user input
if [ "$SCAN_CHOICE" == "yes" ]; then
	scan_and_push_large_files
elif [ "$SCAN_CHOICE" == "no" ]; then
	# Manual file/folder tracking and pushing
	while true; do
		# Ask for the file/folder path
		read -p "Enter the path of the file or folder to track and push (or directly enter to finish): " TARGET_PATH

		# Exit condition
		if [ "$TARGET_PATH" == "" ]; then
			echo "Finished processing all files/folders."
			break
		fi

		# Check if the path exists
		if [ ! -e "$TARGET_PATH" ]; then
			echo "Error: '$TARGET_PATH' does not exist."
			continue
		fi

		# Determine if it's a file or folder
		if [ -f "$TARGET_PATH" ]; then
			echo "Tracking a single file: $TARGET_PATH"
			git lfs track "$TARGET_PATH"
			if [ ! -s "git_lfs_tracked_files.txt" ]; then
				echo "$TARGET_PATH" >>"git_lfs_tracked_files.txt"
			else
				echo -e "\n$TARGET_PATH" >>"git_lfs_tracked_files.txt"
			fi
		elif [ -d "$TARGET_PATH" ]; then
			echo "Tracking all files in folder: $TARGET_PATH"
			git lfs track "$TARGET_PATH/**"
			if [ ! -s "git_lfs_tracked_files.txt" ]; then
				echo "$TARGET_PATH/" >>"git_lfs_tracked_files.txt"
			else
				echo -e "\n$TARGET_PATH/" >>"git_lfs_tracked_files.txt"
			fi
		else
			echo "Error: '$TARGET_PATH' is neither a file nor a folder."
			continue
		fi

		# Stage .gitattributes and the selected file/folder
		git add "$TARGET_PATH"
	done

	# Commit and push to main branch
	# if the tracked files list is empty, remove the file
	if [ ! -s "git_lfs_tracked_files.txt" ]; then
		rm -f "git_lfs_tracked_files.txt"
	else
		git add "git_lfs_tracked_files.txt"
	fi
	git add .gitattributes
	git commit -m "update"
	git push origin "$BRANCH_NAME"
else
	echo "Invalid choice! No action taken."
	exit 0
fi

# post-process: if "git_lfs_tracked_files.txt" exists, check: can I combine those files into the pre-existing folder,
# if so, remove those files, replace with the folder only("add /")
if [ -s "git_lfs_tracked_files.txt" ]; then

	# Create temporary file
	TEMP_FILE="git_lfs_tracked_files.tmp"
	cp "git_lfs_tracked_files.txt" "$TEMP_FILE"

	# Get all unique directories from tracked files
	DIRECTORIES=$(grep -v '/$' "$TEMP_FILE" | xargs -I {} dirname {} | sort -u)

	# For each directory, check if we should consolidate
	for dir in $DIRECTORIES; do
		# Skip current directory
		if [ "$dir" = "." ]; then
			continue
		fi

		# Check if folder entry already exists
		if grep -qxF "$dir/" "$TEMP_FILE"; then
			echo "  ℹ️  Folder already tracked: $dir/"
			continue
		fi

		# Get all files in this directory (from filesystem)
		if [ ! -d "$dir" ]; then
			continue
		fi

		ALL_FILES_IN_DIR=$(find "$dir" -type f -not -path "*/.git/*" 2>/dev/null | sed 's|^\./||')

		if [ -z "$ALL_FILES_IN_DIR" ]; then
			continue
		fi

		# Check if ALL files in directory are tracked
		ALL_TRACKED=true
		while IFS= read -r file_in_dir; do
			if [ -n "$file_in_dir" ]; then
				# Check if this file is in tracking list
				if ! grep -qxF "$file_in_dir" "$TEMP_FILE"; then
					ALL_TRACKED=false
					break
				fi
			fi
		done <<<"$ALL_FILES_IN_DIR"

		# If all files are tracked, consolidate to folder
		if [ "$ALL_TRACKED" = true ]; then
			echo "  ✅ Consolidating:  All files in '$dir/' are tracked"

			# Remove individual file entries for this directory
			grep -v "^$dir/" "$TEMP_FILE" >"${TEMP_FILE}.new"
			mv "${TEMP_FILE}.new" "$TEMP_FILE"

			# Add folder entry
			echo "" >>"$TEMP_FILE"
			echo "$dir/" >>"$TEMP_FILE"
		fi
	done

	# Clean up:  remove empty lines and sort
	sed -i '/^[[:space:]]*$/d' "$TEMP_FILE" 2>/dev/null ||
		sed -i '' '/^[[:space:]]*$/d' "$TEMP_FILE" 2>/dev/null || true

	sort -u "$TEMP_FILE" -o "$TEMP_FILE"

	# Replace original file
	mv "$TEMP_FILE" "git_lfs_tracked_files.txt"

	echo "  📝 Consolidation complete"
fi
