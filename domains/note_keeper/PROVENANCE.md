# How domains/note_keeper was generated

`anneal init` wrote every file in this directory from the interview below. Nothing
here was hand-written; nothing here is hidden.

## The interview

**Name this domain.**

> note keeper

**Describe the job in one or two sentences.**

> Save a short note and read it back when asked.

**Choose what the agent can use.**

> An MCP server

**Give the server's launch command, or its URL.**

> npx -y @modelcontextprotocol/server-filesystem /tmp/anneal-box

**Choose what makes a run right.**

> The answer matches the example

## Tools discovered

- `read_file` (`mcp:note_keeper/read_file`) — Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.
- `read_text_file` (`mcp:note_keeper/read_text_file`) — Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Operates on the file as text regardless of extension. Only works within allowed directories.
- `read_media_file` (`mcp:note_keeper/read_media_file`) — Read a file and return it as a base64-encoded content block with its MIME type. Image and audio files are returned as image/audio content; any other file type is returned as an embedded resource. Only works within allowed directories.
- `read_multiple_files` (`mcp:note_keeper/read_multiple_files`) — Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.
- `write_file` (`mcp:note_keeper/write_file`) — Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.
- `edit_file` (`mcp:note_keeper/edit_file`) — Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.
- `create_directory` (`mcp:note_keeper/create_directory`) — Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.
- `list_directory` (`mcp:note_keeper/list_directory`) — Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.
- `list_directory_with_sizes` (`mcp:note_keeper/list_directory_with_sizes`) — Get a detailed listing of all files and directories in a specified path, including sizes. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is useful for understanding directory structure and finding specific files within a directory. Only works within allowed directories.
- `directory_tree` (`mcp:note_keeper/directory_tree`) — Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.
- `move_file` (`mcp:note_keeper/move_file`) — Move or rename files and directories. Can move files between directories and rename them in a single operation. If the destination exists, the operation will fail. Works across different directories and can be used for simple renaming within the same directory. Both source and destination must be within allowed directories.
- `search_files` (`mcp:note_keeper/search_files`) — Recursively search for files and directories matching a pattern. The patterns should be glob-style patterns that match paths relative to the working directory. Use pattern like '*.ext' to match files in current directory, and '**/*.ext' to match files in all subdirectories. Returns full paths to all matching items. Great for finding files when you don't know their exact location. Only searches within allowed directories.
- `get_file_info` (`mcp:note_keeper/get_file_info`) — Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.
- `list_allowed_directories` (`mcp:note_keeper/list_allowed_directories`) — Returns the list of directories that this server is allowed to access. Subdirectories within these allowed directories are also accessible. Use this to understand which directories and their nested paths are available before trying to access files.

## Examples and splits

- 3 examples given, split seeded with seed 0: holdout 1, search 1, train 1
- success criterion: `examples` (evaluator template used for eval.py)
- goal.md source: `template` (`template` = deterministic only; `elaborated` = model-written rules appended)

> Only 3 examples: enough to run, too few for the reserved evaluation split to mean anything. Add more before quoting a number (see PROVENANCE.md).
