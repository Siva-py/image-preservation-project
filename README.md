# ImagePreserve

**A Multi-Stage Framework for Quality Filtering and Information-Preserving Image Dataset Reduction**

ImagePreserve is a project focused on reducing image dataset size while trying to preserve useful visual information. It brings image preprocessing, quality filtering, and information-preservation steps together in a modular pipeline.

The project is being developed as part of a dissertation. The aim is to make image datasets easier to manage without treating every image as equally useful or applying reduction blindly.

## What the project does

The application provides a web interface for uploading images, configuring the pipeline, running processing through a FastAPI backend, and viewing the results.

The pipeline is designed to support these main stages:

- **Image preprocessing and compression** to reduce image file sizes.
- **Quality filtering** to identify images that do not meet the project's quality criteria.
- **Duplicate and near-duplicate handling** to help reduce repeated visual content.
- **Information-preservation and dataset selection** to retain a chosen portion of the usable dataset.
- **Results and reporting** to review the output of a completed run.

The exact processing steps depend on the modules currently integrated into the project.

## Project structure

The repository is organized into separate frontend and backend parts. A simplified view is shown below; some folders may contain additional files as development continues.

```text
image-preservation-project/
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
│       ├── carousel/
│       └── friends/
├── src/
│   ├── main.py
│   └── api/
│       ├── routes.py
│       ├── carousel_routes.py
│       └── easter_egg_routes.py
├── .gitignore
└── README.md
```

### Frontend

The frontend contains the web interface:

- `index.html` defines the page layout.
- `style.css` controls the visual design and responsive styling.
- `script.js` handles navigation, carousel behaviour, image uploads, pipeline requests, progress feedback, galleries, and dashboard reporting.
- `assets/carousel/` contains images used in the normal homepage carousel.
- `assets/friends/` contains the photos used by the optional friends-carousel Easter egg.

### Backend

The backend uses FastAPI to expose API endpoints to the frontend.

- `src/main.py` creates the FastAPI application, configures middleware, and registers routers.
- `src/api/routes.py` contains the main application routes.
- `src/api/carousel_routes.py` handles the regular homepage carousel.
- `src/api/easter_egg_routes.py` provides the endpoint that lists photos in the friends folder.

The API routes are kept separate so features can be maintained without putting all of the application logic in one file.

## Requirements

- Python 3.10 or a compatible version supported by the project's dependencies
- pip
- A modern web browser
- The project's Python dependencies installed in a virtual environment

The exact dependencies should be taken from the dependency file maintained in the repository, if one is present.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd image-preservation-project
```

Replace `<repository-url>` with the repository's Git URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, follow your machine's Python environment policy or activate the environment using the method configured on your system.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If dependencies are managed through another project file, use that file's installation instructions.

## Running the application

### 1. Start the backend

From the project root, with the virtual environment activated, run:

```bash
python -m src.main
```

If the project is configured to run through Uvicorn directly, this is also a common development command:

```bash
uvicorn src.main:app --reload
```

Use the command that matches the current project setup. Do not run both commands at the same time on the same port.

The backend's default address in the frontend is:

```text
http://127.0.0.1:8000
```

You can check that the API is running by opening:

```text
http://127.0.0.1:8000/
```

The API documentation is normally available at:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the frontend

Open the `frontend` folder using a local development server, such as the VS Code Live Server extension. For example, Live Server commonly uses:

```text
http://127.0.0.1:5500
```

Keep the backend running while using the frontend. Opening the HTML file directly with `file://` may cause browser restrictions with API requests or local assets, so a local web server is recommended.

If the backend is running at a different address or port, update `API_BASE_URL` in `frontend/script.js` to match it.

## Using the application

1. Open the frontend in your browser.
2. Go to **Process Images**.
3. Select or drag and drop the images you want to process.
4. Configure the available processing options.
5. Start the pipeline and watch the progress message.
6. When processing finishes successfully, review the results and image galleries.
7. Export the dashboard report or download the final dataset if those options are enabled for the completed run.

Processing options and available results depend on which backend modules are currently connected.

## Homepage carousel

The homepage includes a carousel that loads its images through the backend. The regular carousel images are stored in:

```text
frontend/assets/carousel/
```

Keep the folder name consistent with the path expected by the existing carousel route.

## Friends-carousel Easter egg

There is also a small Easter egg: clicking the ImagePreserve logo switches the homepage carousel to photos stored in the friends folder. Clicking the logo again returns to the normal carousel.

Add photos here:

```text
frontend/assets/friends/
```

Supported image formats include JPG, JPEG, PNG, WEBP, BMP, and GIF.

The frontend requests the list of filenames from this backend endpoint:

```text
GET /api/easter-egg/friends
```

The route scans the folder when requested, so you do not need to maintain a separate JSON file listing the photos. Newly added or removed photos are reflected the next time friends mode is activated, provided the frontend can access the corresponding image files.

If the Easter egg does not work, check that:

- `src/api/easter_egg_routes.py` exists.
- Its router is registered in `src/main.py` with the `/api` prefix.
- The backend is running.
- The photos are inside `frontend/assets/friends/`.
- The frontend is being served from a local web server and can load files from `assets/friends/`.

## API and troubleshooting

### The frontend cannot connect to FastAPI

- Make sure the backend is running.
- Check that `API_BASE_URL` in `frontend/script.js` matches the backend address.
- Check the browser console and the terminal running FastAPI for errors.
- Confirm that the frontend's local-server origin is allowed by the backend's CORS configuration.

### The carousel is empty

- Check that the correct images are in the expected folder.
- Confirm the regular carousel endpoint is registered.
- Open the browser developer tools and inspect the Network and Console tabs for failed requests.

### The friends carousel is empty

- Confirm that the friends route returns image filenames at `/api/easter-egg/friends`.
- Make sure the folder contains supported image files.
- Check that the image URLs work from the frontend's local-server address.

## Working with Git

Before committing changes, check the files you are about to add:

```bash
git status
```

Commit your work with a clear message, for example:

```bash
git add README.md
git commit -m "Add project README"
git push
```

Do not commit virtual environments, temporary outputs, private datasets, or large generated files unless the project specifically requires them. Check `.gitignore` before pushing.

## Project status

ImagePreserve is an ongoing project. The frontend, API routes, and processing modules may evolve as the pipeline is developed and evaluated. Refer to the current source code and dependency files for the latest implementation details.

## Contributors

- **Siva D M**
- **Bhagavadh B**
