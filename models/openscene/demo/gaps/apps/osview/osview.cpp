// Source file for the open vocabulary viewer program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

namespace gaps {}
using namespace gaps;
#include "R3Graphics/R3Graphics.h"
#include "R3Surfels/R3Surfels.h"
#include "RNMath/RNMath.h"
#include "RNNets/RNNets.h"
#include "RGBD/RGBD.h"
#include "fglut/fglut.h"
#include "half.hpp"
#include "npy.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static RNArray<const char *> input_mesh_filenames;
static RNArray<const char *>input_ssa_filenames;
static RNArray<const char *> input_ssb_filenames;
static RNArray<const char *> input_point_features_filenames;
static RNArray<const char *> input_configuration_filenames;
static const char *input_category_names_filename = NULL;
static const char *input_category_colors_filename = NULL;
static const char *input_category_features_filename = NULL;
static const char *input_scene_filename = NULL;
static const char *input_image_directory = NULL;
static RNBoolean one_feature_vector_per_object = FALSE;
static RNInterval default_value_range(0.05,0.1);
static R3Box scene_extent(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
static R3Box viewing_extent(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
static int use_tcp = 0;
static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////

enum {
  RGB_COLOR,
  FEATURE_COLOR,
  AFFINITY_COLOR,
  SEGMENTATION_COLOR,
  OVERLAY_COLOR,
  PICK_COLOR,
  NUM_COLOR_SCHEMES
};



////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////

#if 1
#define RNDenseMatrix RNFloatMatrix

struct RNFloatMatrix {
public:
  RNFloatMatrix(int nrows = 0, int ncolumns = 0)
    : nrows(nrows), ncolumns(ncolumns), values()
    { if (nrows * ncolumns > 0) values.resize(nrows * ncolumns); }
  RNFloatMatrix(const RNFloatMatrix& m)
    : nrows(m.nrows), ncolumns(m.ncolumns), values()
    { if (nrows * ncolumns > 0) {
        values.resize(nrows * ncolumns);
        for (int i = 0; i < nrows * ncolumns; i++) values[i] = m.values[i];
      }
    }
  int NRows(void) const
    { return nrows; }
  int NColumns(void) const
    { return ncolumns; }
  float *operator[](int k)
    { return &values[k*ncolumns]; }
  const float *operator[](int k) const
    { return &values[k*ncolumns]; }
public:
  int nrows;
  int ncolumns;
  std::vector<float> values;
};

struct RNHalfMatrix {
public:
  RNHalfMatrix(int nrows = 0, int ncolumns = 0)
    : nrows(nrows), ncolumns(ncolumns), values()
    { if (nrows * ncolumns > 0) values.resize(nrows * ncolumns); }
  RNHalfMatrix(const RNHalfMatrix& m)
    : nrows(m.nrows), ncolumns(m.ncolumns), values()
    { if (nrows * ncolumns > 0) {
        values.resize(nrows * ncolumns);
        for (int i = 0; i < nrows * ncolumns; i++) values[i] = m.values[i];
      }
    }
  int NRows(void) const
    { return nrows; }
  int NColumns(void) const
    { return ncolumns; }
  half_float::half *operator[](int k)
    { return &values[k*ncolumns]; }
public:
  int nrows;
  int ncolumns;
  std::vector<half_float::half> values;
};

#endif



////////////////////////////////////////////////////////////////////////
// Internal variables
////////////////////////////////////////////////////////////////////////

// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_width = 1024;
static int GLUTwindow_height = 768;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmouse_drag = 0;
static int GLUTmodifiers = 0;


// Data variables

static RNArray<R3Mesh *> meshes;
static RNArray<R3SurfelScene *> surfels;
static RNArray<RNDenseMatrix *> point_features;
static RNArray<RNVector *> mesh_affinities;
static RNArray<RNVector *> mesh_segmentations;
static RNDenseMatrix *category_features = NULL;
static RNDenseMatrix *category_colors = NULL;
static RNArray<char *> *category_names = NULL;
static R3Scene *scene = NULL;


// Interaction variables

static std::string query_string;
static RNVector query_features;
static R3Point selected_position(-1, -1, -1);
static int selected_category_index = 0;
static R3SurfelImage *selected_image = NULL;
static char *screenshot_image_name = NULL;
static int color_scheme = OVERLAY_COLOR;
static R2Image inset_image_pixels;
static double inset_image_size = 0.2;
static RNScalar max_affinity = 0;
static RNInterval value_range(default_value_range);
static RNRgb background(0,0,0);
static R3Point center(0, 0, 0);
static R3Viewer viewer;


// Display variables

static int show_query_string = 1;
static int show_cameras = 1;
static int show_scene = 0;
static int show_faces = 1;
static int show_vertices = 0;
static int show_inset_image = 1;
static int show_category_names = 0;
static int show_selected_position = 1;
static int show_weak_affinities = 1;
static int show_axes = 0;


// VBO variables

static GLuint vbo_point_position_buffer = 0;
static GLuint vbo_point_normal_buffer = 0;
static GLuint vbo_point_color_buffer = 0;
static GLuint vbo_face_index_buffer = 0;
static unsigned int vbo_nvertices = 0;
static unsigned int vbo_nfaces = 0;
static int vbo_color_scheme = -1;


// Query variables

std::string query_feature_generator("python3 ~/gaps/apps/osview/generate_one_clip_feat.py");
std::string query_feature_directory("tmp");


// Pick variables

static unsigned char CAMERA_ALPHA = 254;



////////////////////////////////////////////////////////////////////////
// Read/Write functions
////////////////////////////////////////////////////////////////////////

static R3Mesh *
ReadMeshFile(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    RNFail("Unable to allocate mesh for %s\n", filename);
    return 0;
  }

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    RNFail("Unable to read mesh from %s\n", filename);
    return 0;
  }

  // Set vertex values (indicating row of point features)
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    mesh->SetVertexValue(vertex, i);
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return mesh
  return mesh;
}



static R3SurfelScene *
OpenSurfelsFiles(const char *scene_name, const char *database_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene
  R3SurfelScene *scene = new R3SurfelScene();
  if (!scene) {
    RNFail("Unable to allocate scene\n");
    return NULL;
  }

  // Open scene files
  if (!scene->OpenFile(scene_name, database_name, "r", "r")) {
    delete scene;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Opened surfel scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Object Properties = %d\n", scene->NObjectProperties());
    printf("  # Label Properties = %d\n", scene->NLabelProperties());
    printf("  # Object Relationships = %d\n", scene->NObjectRelationships());
    printf("  # Label Relationships = %d\n", scene->NLabelRelationships());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Scans = %d\n", scene->NScans());
    printf("  # Images = %d\n", scene->NImages());
    printf("  # Nodes = %d\n", scene->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene->Tree()->NBlocks());
    printf("  # Surfels = %lld\n", scene->Tree()->NSurfels());
    fflush(stdout);
  }

  // Return scene
  return scene;
}



static int
CloseSurfelsFiles(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Print statistics
  if (print_verbose) {
    printf("Closing scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Scans = %d\n", scene->NScans());
    printf("  # Images = %d\n", scene->NImages());
    printf("  # Nodes = %d\n", scene->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene->Tree()->NBlocks());
    printf("  # Surfels = %lld\n", scene->Tree()->NSurfels());
    fflush(stdout);
  }

  // Close scene files
  if (!scene->CloseFile()) {
    delete scene;
    return 0;
  }

  // Return success
  return 1;
}



static R3Mesh *
CreateMeshFromSurfels(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    RNFail("Unable to allocate mesh\n");
    return NULL;
  }

  // Load surfels into mesh
  R3SurfelTree *tree = scene->Tree();
  R3SurfelDatabase *database = tree->Database();
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    if (node->NParts() > 0) continue;
    R3SurfelObject *object = node->Object(TRUE, TRUE);
    // while (object && object->Parent() && (object->Parent() != scene->RootObject()))  object = object->Parent();
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      database->ReadBlock(block);
      for (int k = 0; k < block->NSurfels(); k++) {
        R3Point position = block->SurfelPosition(k);
        R3Vector normal = block->SurfelNormal(k);
        RNRgb color = block->SurfelColor(k);
        int index = block->SurfelIdentifier(k);
        if (one_feature_vector_per_object) index = (object) ? object->SceneIndex() : scene->NObjects();
        R3MeshVertex *vertex = mesh->CreateVertex(position, normal, color);
        mesh->SetVertexValue(vertex, index);
      }
      database->ReleaseBlock(block);
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Created mesh from surfels ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return mesh
  return mesh;
}



static R3Scene *
ReadSceneFile(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene
  R3Scene *scene = new R3Scene();
  if (!scene) {
    RNFail("Unable to allocate scene for %s\n", filename);
    return 0;
  }

  // Read scene from file
  if (!scene->ReadFile(filename)) {
    RNFail("Unable to read scene from %s\n", filename);
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read scene from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", scene->NNodes());
    fflush(stdout);
  }

  // Return scene
  return scene;
}



static R3SurfelScene *
ReadConfigurationFile(const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Reading configuration from %s ...\n", filename);
    fflush(stdout);
  }

  // Read file
  RGBDConfiguration configuration;
  if (!configuration.ReadFile(filename)) {
    RNFail("Unable to read configuration from %s\n", filename);
    return NULL;
  }

  // Allocate surfel scene
  R3SurfelScene *surfels = new R3SurfelScene();
  if (!surfels) {
    RNFail("Unable to allocate surfel scene for %s\n", filename);
    return NULL;
  }

  // Create surfel images from RGBD images
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *rgbd_image = configuration.Image(i);

    // Get name
    char name[1025];
    if (rgbd_image->Name()) strncpy(name, rgbd_image->Name(), 1024);
    else sprintf(name, "Image_%d\n", i);

    // Get stuff from rgbd image
    R3Point viewpoint = rgbd_image->WorldViewpoint();
    R3Vector towards = rgbd_image->WorldTowards();
    R3Vector up = rgbd_image->WorldUp();
    int width = rgbd_image->NPixels(RN_X);
    int height = rgbd_image->NPixels(RN_Y);
    R3Matrix intrinsics = rgbd_image->Intrinsics();
    double xfocal = intrinsics[0][0];
    double yfocal = intrinsics[1][1];
    double xcenter = intrinsics[0][2];
    double ycenter = intrinsics[1][2];

    // Create sfl image
    R3SurfelImage *sfl_image = new R3SurfelImage();
    sfl_image->SetViewpoint(viewpoint);
    sfl_image->SetOrientation(towards, up);
    sfl_image->SetImageDimensions(width, height);
    sfl_image->SetImageCenter(R2Point(xcenter, ycenter));
    sfl_image->SetXFocal(xfocal);
    sfl_image->SetYFocal(yfocal);
    sfl_image->SetName(name);

    // Insert sfl image into surfel scene
    surfels->InsertImage(sfl_image);
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", surfels->NImages());
    fflush(stdout);
  }

  // Return surfel scene
  return surfels;
}



static RNArray<char *> *
ReadCategoryNamesFile(const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate array of category names
  RNArray<char *> *names = new RNArray<char *>();
  if (!names) {
    fprintf(stderr, "Unable to allocate array for category names\n");
    return 0;
  }

  // Check file type
  if (strstr(filename, ".npy")) {
    // Read names npy file
    unsigned char *array = NULL;
    int data_type, data_size, fortran_order, width, height, depth;
    if (!ReadNumpyFile(filename, &data_type, &data_size, &fortran_order, &width, &height, &depth, &array)) {
      fprintf(stderr, "Unable to read npy file %s\n", filename);
      delete names;
      return 0;
    }

    // Check height (shoud be 1)
    if (height != 1) {
      fprintf(stderr, "Unrecognized shape in %s\n", filename);
      if (array) delete [] array;
      delete names;
      return 0;
    }

    // Check depth (shoud be 1)
    if (depth != 1) {
      fprintf(stderr, "Unrecognized shape in %s\n", filename);
      if (array) delete [] array;
      delete names;
      return 0;
    }

    // Fill category names
    unsigned char *arrayp = array;
    for (int i = 0; i < width; i++) {
      char name[4096];
      if (data_size > 4095) data_size = 4095;
      for (int j = 0; j < data_size; j++) {
        name[j] = *arrayp++;
        if (data_type == 'U') arrayp += 3;
      }
      name[data_size] = '\0';
      names->Insert(strdup(name));
    }
  }
  else {
    // Open names file
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      RNFail("Unable to open category names file %s\n", filename);
      delete names;
      return NULL;
    }
  
    // Read names file
    char buffer[1024];
    while (fgets(buffer, 1023, fp)) {
      buffer[strcspn(buffer, "\r\n")] = '\0';
      char *name = strdup(buffer);
      names->Insert(name);
    }
  
    // Close names file
    fclose(fp);
  }
  
  // Print statistics
  if (print_verbose) {
    printf("Read category names from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Names = %d\n", names->NEntries());
    if (print_debug) {
      for (int i = 0; i < names->NEntries(); i++) {
        printf("    %d %s\n", i, names->Kth(i));
      }
    }
    fflush(stdout);
  }

  // Return category names
  return names;
}




static RNDenseMatrix *
ReadCategoryColorsFile(const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();


  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    RNFail("Unable to open category colors file %s\n", filename);
    return NULL;
  }
  
  // Read file
  char buffer[1024];
  std::vector<RNRgb> rgbs;
  while (fgets(buffer, 1023, fp)) {
    double r, g, b;
    if (sscanf(buffer, "%lf%lf%lf", &r, &g, &b) != (unsigned int) 3) {
      RNFail("Error reading entry %d of category color file %s\n", rgbs.size());
      return NULL;
    }
    rgbs.push_back(RNRgb(r, g, b));
  }
  
  // Close file
  fclose(fp);

  // Allocate matrix of category colors
  RNDenseMatrix *colors = new RNDenseMatrix(rgbs.size(), 3);
  if (!colors) {
    fprintf(stderr, "Unable to allocate matrix for category colors\n");
    return 0;
  }
  
  // Fill matrix with category colors
  for (unsigned int i = 0; i < rgbs.size(); i++) {
    (*colors)[i][0] = rgbs[i][0];
    (*colors)[i][1] = rgbs[i][1];
    (*colors)[i][2] = rgbs[i][2];
  }
  
  // Print statistics
  if (print_verbose) {
    printf("Read category colors from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Colors = %d\n", colors->NRows());
    fflush(stdout);
  }

  // Return category colors
  return colors;
}



static RNDenseMatrix *
ReadFeaturesFile(const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read features npy file
  unsigned char *array = NULL;
  int data_type, data_size, fortran_order, width, height, depth;
  if (!ReadNumpyFile(filename, &data_type, &data_size, &fortran_order, &width, &height, &depth, &array)) {
    fprintf(stderr, "Unable to read npy file %s\n", filename);
    return 0;
  }

  // Check depth (shoud be 1)
  if (depth != 1) {
    fprintf(stderr, "Unrecognized shape in %s\n", filename);
    if (array) delete [] array;
    return 0;
  }

  // Create matrix of features
  RNDenseMatrix *matrix = new RNDenseMatrix(width, height);
  if (!matrix) {
    fprintf(stderr, "Unable to allocate matrix for %s\n", filename);
    if (array) delete [] array;
    return 0;
  }
  
  // Fill matrix of features
  unsigned char *arrayp = array;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      if (data_size == 2) {
        half_float::half *a = (half_float::half *) arrayp;
        (*matrix)[i][j] = (float) *a;
        arrayp += 2;
      }
      else if (data_size == 4) {
        float *a = (float *) arrayp;
        (*matrix)[i][j] = *a;
        arrayp += 4;
      }
      else if (data_size == 8) {
        double *a = (double *) arrayp;
        (*matrix)[i][j] = *a;
        arrayp += 8;
      }
    }
  }

  // Normalize rows
  for (int i = 0; i < matrix->NRows(); i++) {
    RNScalar sum = 0;
    for (int j = 0; j < matrix->NColumns(); j++) {
      RNScalar value = (*matrix)[i][j];
      sum += value * value;
    }
    if (sum == 0) continue;
    RNScalar denom = sqrt(sum);    
    for (int j = 0; j < matrix->NColumns(); j++) {
      (*matrix)[i][j] /= denom;
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Read features from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Entries = %d\n", matrix->NRows());
    printf("  # Features = %d\n", matrix->NColumns());
    fflush(stdout);
  }

  // Return matrix of category features
  return matrix;
}



///////////////////////////////////////////////////////////////////////
// Color functions
////////////////////////////////////////////////////////////////////////

static RNScalar
NormalizedValue(RNScalar value)
{
  // Return value between 0 and 1
  RNScalar diameter = value_range.Diameter();
  if (diameter > 0) value = (value - value_range.Min()) / diameter;
  if (value < 0) value = 0;
  else if (value > 1) value = 1;
  return value;
}



static RNRgb 
IndexedColor(int index)
{
  // Return unique color for index
  index++;
  static const double s = 1.0 / 255.0;
  char r = (index * 191) % 256;
  char g = (index * 103) % 256;
  char b = (index * 51) % 256;
  return RNRgb(s*r, s*g, s*b);
}



static RNRgb 
CategoryColor(int index)
{
  // Return color representing category
  if (!category_colors || (index >= category_colors->NRows())) return IndexedColor(index);
  else return RNRgb((*category_colors)[index][0], (*category_colors)[index][1], (*category_colors)[index][2]);
}



static RNRgb 
NormalizedColor(RNScalar value, int color_scheme)
{
  // Check color scheme
  RNRgb c(0, 0, 0);

  // Normalize value
  value = 1.0 - NormalizedValue(value);

  if (0 && (color_scheme == OVERLAY_COLOR)) {
    // Compute color blue-to-red
    if (value < 0.5) {
      c[0] = 1 - 2 * value;
      c[1] = 2 * value;
     }
     else {
      c[1] = 1 - 2 * (value - 0.5);
      c[2] = 2 * (value - 0.5);
    }
  }
  else {
    // Compute color blue-to-yellow
    if (value < 0.5) {
      c[0] = 1 - 2 * value;
      c[1] = 1;
     }
     else {
      c[1] = 1 - 2 * (value - 0.5);
      c[2] = 2 * (value - 0.5);
    }
  }

  // Return color
  return c;
}



static RNRgb
ComputeColor(RNDenseMatrix *features, RNVector *affinities, RNVector *segmentation, int index, const RNRgb& rgb, int color_scheme)
{
  // Check the color scheme
  if (color_scheme == RGB_COLOR) {
    return rgb;
  }
  else if ((color_scheme == FEATURE_COLOR) && features) {
#if 1
    RNScalar r = fabs((*features)[index][0]);
    RNScalar g = fabs((*features)[index][1]);
    RNScalar b = fabs((*features)[index][2]);
#else
    int step = 1; // features->NColumns()/30;
    RNScalar r = 0, g = 0, b = 0;
    for (int i = 0; i < features->NColumns(); i += step) {
      RNScalar f = (*features)[index][i];
      if  (i < features->NColumns()/3) r += f;
      else if  (i < 2*features->NColumns()/3) g += f;
      else b += f;
    }
#endif
    r = r*r; g = g*g;  b= b*b;
    RNScalar sum = r + g + b;
    if (sum <= 0) return RNwhite_rgb; 
    RNScalar scale = 1.0 / sum;
    return RNRgb(scale * r, scale * g, scale * b);
  }
  else if ((color_scheme == AFFINITY_COLOR) && affinities) {
    RNScalar value = (*affinities)[index];
    return NormalizedColor(value, color_scheme);
  }
  else if (color_scheme == SEGMENTATION_COLOR && segmentation) {
    int category = (*segmentation)[index] + 0.5;
    return CategoryColor(category);
  }
  else if (color_scheme == OVERLAY_COLOR) {
    RNScalar value = (affinities) ? (*affinities)[index] : 0;
    if (value > value_range.Min()) return NormalizedColor(value, color_scheme);
    else return rgb;
  }
  else if (color_scheme == PICK_COLOR) {
    return RNgray_rgb;
  }
  else {
    return RNgray_rgb;
  }
}



////////////////////////////////////////////////////////////////////////
// Viewing extent utility functions
////////////////////////////////////////////////////////////////////////

static void 
DrawViewingExtent(void)
{
  // Check viewing extent
  if (viewing_extent.IsEmpty()) return;
  
  // Draw viewing extent
  glDisable(GL_LIGHTING);
  RNLoadRgb(0.5, 0.5, 0.5);
  viewing_extent.Outline();
}

  

static void 
DisableViewingExtent(void)
{
  // Disable all clip planes
  for (int i = 0; i < 6; i++) {
    glDisable(GL_CLIP_PLANE0 + i);
  }
}



static void 
EnableViewingExtent(void)
{
  // Check viewing extent
  if (viewing_extent.IsEmpty() || R3Contains(viewing_extent, scene_extent)) {
    DisableViewingExtent();
    return;
  }

  // Load lo clip planes
  for (int dim = RN_X; dim <= RN_Z; dim++) {
    if (viewing_extent[RN_LO][dim] > scene_extent[RN_LO][dim]) {
      GLdouble plane_equation[4] = { 0, 0, 0, 0 };
      plane_equation[dim] = 1.0;
      plane_equation[3] = -viewing_extent[RN_LO][dim];
      glClipPlane(GL_CLIP_PLANE0 + dim, plane_equation);
      glEnable(GL_CLIP_PLANE0 + dim);
    }
  }

  // Load hi clip planes
  for (int dim = RN_X; dim <= RN_Z; dim++) {
    if (viewing_extent[RN_HI][dim] < scene_extent[RN_HI][dim]) {
      GLdouble plane_equation[4] = { 0, 0, 0, 0 };
      plane_equation[dim] = -1.0;
      plane_equation[3] = viewing_extent[RN_HI][dim];
      glClipPlane(GL_CLIP_PLANE0 + 3 + dim, plane_equation);
      glEnable(GL_CLIP_PLANE0 + 3 + dim);
    }
  }
}  



////////////////////////////////////////////////////////////////////////
// VBO management functions
////////////////////////////////////////////////////////////////////////

static int
IsFaceVisible(R3Mesh *mesh, R3MeshFace *face, int m)
{
  // Check affinities
  if (!show_weak_affinities) {
    for (int i = 0; i < 3; i++) {
      R3MeshVertex *vertex = mesh->VertexOnFace(face, i);
      int index = mesh->VertexValue(vertex) + 0.5;      
      RNScalar affinity = (*mesh_affinities[m])[index];
      if (affinity < value_range.Min()) return 0;
    }
  }
    
  // Passed all tests
  return 1;
}


  
static void
UpdateVertexVBO(int color_scheme)
{
  // Check if VBO is uptodate
  if (vbo_nvertices > 0) return;
  
  // Count points
  vbo_nvertices = 0;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes.Kth(m);
    vbo_nvertices += mesh->NVertices();
  }

  // Check point count
  if (vbo_nvertices <= 0) return;

  // Allocate in-memory buffers
  GLfloat *point_positions = new GLfloat [ 3 * vbo_nvertices ];
  GLfloat *point_normals = new GLfloat [ 3 * vbo_nvertices ];
  GLubyte *point_colors = new GLubyte [  3 * vbo_nvertices ];

  // Fill buffers from mesh vertices
  GLfloat *point_positionsp = point_positions;
  GLfloat *point_normalsp = point_normals;
  GLubyte *point_colorsp = point_colors;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes.Kth(m);
    RNDenseMatrix *features = (m < point_features.NEntries()) ? point_features[m] : NULL;
    RNVector *affinities = (m < mesh_affinities.NEntries()) ? mesh_affinities[m] : NULL;
    RNVector *segmentation = (m < mesh_segmentations.NEntries()) ? mesh_segmentations[m] : NULL;
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      int index = mesh->VertexValue(vertex) + 0.5;
      const R3Point& position = mesh->VertexPosition(vertex);
      const R3Vector& normal = mesh->VertexNormal(vertex);
      const RNRgb& rgb = mesh->VertexColor(vertex);
      RNRgb color = ComputeColor(features, affinities, segmentation, index, rgb, color_scheme);
      *(point_positionsp++) = position.X();
      *(point_positionsp++) = position.Y();
      *(point_positionsp++) = position.Z();
      *(point_normalsp++) = normal.X();
      *(point_normalsp++) = normal.Y();
      *(point_normalsp++) = normal.Z();
      *(point_colorsp++) = 255.0 * color.R();
      *(point_colorsp++) = 255.0 * color.G();
      *(point_colorsp++) = 255.0 * color.B();
    }
  }
  
  // Just checking
  assert(point_positionsp - point_positions == 3*vbo_nvertices);
  assert(point_normalsp - point_normals == 3*vbo_nvertices);
  assert(point_colorsp - point_colors == 3*vbo_nvertices);

  // Generate VBO buffers (first time only)
  if (vbo_point_position_buffer == 0) glGenBuffers(1, &vbo_point_position_buffer);
  if (vbo_point_normal_buffer == 0) glGenBuffers(1, &vbo_point_normal_buffer);
  if (vbo_point_color_buffer == 0) glGenBuffers(1, &vbo_point_color_buffer);

  // Load VBO buffers
  if (vbo_point_position_buffer && point_positions) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point_position_buffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * vbo_nvertices * sizeof(GLfloat), point_positions, GL_STATIC_DRAW);
    glVertexPointer(3, GL_FLOAT, 0, 0);
  }
  if (vbo_point_normal_buffer && point_normals) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point_normal_buffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * vbo_nvertices * sizeof(GLfloat), point_normals, GL_STATIC_DRAW);
    glNormalPointer(GL_FLOAT, 0, 0);
  }
  if (vbo_point_color_buffer && point_colors) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point_color_buffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * vbo_nvertices * sizeof(GLubyte), point_colors, GL_STATIC_DRAW);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
  }

  // Delete in-memory buffers
  if (point_positions) delete [] point_positions;
  if (point_normals) delete [] point_normals;
  if (point_colors) delete [] point_colors;
}



static void
UpdateFaceVBO(int color_scheme)
{
  // Check if VBO is uptodate
  if (vbo_nfaces > 0) return;

  // Count faces
  vbo_nfaces = 0;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes.Kth(m);
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);
      if (!IsFaceVisible(mesh, face, m)) continue;
      vbo_nfaces++;
    }
  }

  // Check face count
  if (vbo_nfaces == 0) return;

  // Allocate in-memory buffers
  GLint *face_index = new GLint [ 3 * vbo_nfaces ];

  // Fill in-memory face buffers 
  GLint *face_indexp = face_index;
  int offset = 0;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes.Kth(m);
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);
      if (!IsFaceVisible(mesh, face, m)) continue;
      for (int j = 0; j < 3; j++) {
        R3MeshVertex *vertex = mesh->VertexOnFace(face, j);
        int index = offset + mesh->VertexID(vertex);
        *(face_indexp++) = index;
      }
    }
    offset += mesh->NVertices();
  }
  
  // Just checking
  assert(face_indexp - face_index == 3*vbo_nfaces);

  // Generate VBO buffer (first time only)
  if (vbo_face_index_buffer == 0) glGenBuffers(1, &vbo_face_index_buffer);

  // Load VBO buffers
  if (vbo_face_index_buffer && face_index) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_face_index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * vbo_nfaces * sizeof(unsigned int), face_index, GL_STATIC_DRAW);
  }

  // Delete in-memory buffers
  if (face_index) delete [] face_index;
}



static void 
InvalidateVBO(void)
{
  // Mark mesh VBOs as out of date
  vbo_nvertices = 0;
  vbo_nfaces = 0;

}



////////////////////////////////////////////////////////////////////////
// Inset image management functions
////////////////////////////////////////////////////////////////////////

#if 0
static R2Image
ComputeSegmentationImage(R3SurfelImage *image)
{
  // Get/check stuff
  if (!input_image_directory) return R2Image();
  if (!image) return R2Image();
  if (!image->Name()) return R2Image();
  int image_width = image->ImageWidth();
  int image_height = image->ImageHeight();
  if ((image_width <= 0) || (image_height <= 0)) return R2Image();
  if (!category_features) return R2Image();

  // Read image features from file
  unsigned char *values = NULL;
  int data_type = 0, data_size = 0, fortran_order = 0, width = 0, height = 0, depth = 0;
  char filename[1024];
  sprintf(filename, "%s/clip_image_features/%s.npy", input_image_directory, image->Name());
  if (!ReadNumpyFile(filename, &data_type, &data_size,
    &fortran_order, &width, &height, &depth, &values)) {
    fprintf(stderr, "Unable to read npy file %s\n", filename);
    return R2Image();
  }

  // Determine scaling from surfel image to feature image
  double xscale = (double) height / (double) image_width;
  double yscale = (double) width  / (double) image_height;

  // Compute segmentantion image with category colors
  R2Image segmentation_image(image_width, image_height, 3);
  for (int iy = 0; iy < image_height; iy++) {
    int features_iy = yscale * (image_height - iy - 1) + 0.5;
    if ((features_iy < 0) || (features_iy >= width)) continue;
    for (int ix = 0; ix < image_width; ix++) {
      int features_ix = xscale * ix + 0.5;
      if ((features_ix < 0) || (features_ix >= height)) continue;
      unsigned char *pixel_featuresp = values + 2 * depth * ((features_iy * height) + features_ix);
      half_float::half *pixel_features = (half_float::half *) pixel_featuresp;
      RNScalar ssum = 0;
      for (int k = 0; k < depth; k++) ssum += pixel_features[k] * pixel_features[k];
      RNScalar l2norm = sqrt(ssum);
      int best_category_index = -1;
      RNScalar best_category_affinity = -FLT_MAX;
      for (int category_index = 0; category_index < category_features->NRows(); category_index++) {
        RNScalar affinity = 0;
        for (int k = 0; k < depth; k++) affinity += (*category_features)[category_index][k] * pixel_features[k] / l2norm;
        if (affinity > best_category_affinity) {
          best_category_index = category_index;
          best_category_affinity = affinity;
        }
      }
      if (best_category_index >= 0) {
        RNRgb color = CategoryColor(best_category_index);
        segmentation_image.SetPixelRGB(ix, iy, color);
      }
    }
  }

  // Delete feature image
  delete [] values;

  // Return segmentation image
  return segmentation_image;
}
#endif



static void
UpdateInsetImage(int color_scheme)
{
  // Static state variables
  static int previous_color_scheme = -1;
  static R3SurfelImage *previous_selected_image = NULL;

  // Get/check stuff
  if (!selected_image) return;
  if (!selected_image->Name()) return;
  int image_width = selected_image->ImageWidth();
  int image_height = selected_image->ImageHeight();
  if ((image_width <= 0) || (image_height <= 0)) return;

  // Check if everything is already uptodate
  if ((selected_image == previous_selected_image) && (color_scheme == previous_color_scheme)) return;
  previous_color_scheme = color_scheme;
  previous_selected_image = selected_image;

  // Read segmentation image, if can
  if ((color_scheme == SEGMENTATION_COLOR) && input_image_directory) {
    // Read color image from file
    char filename[1024];
    sprintf(filename, "%s/clip_category_images/%s.png", input_image_directory, selected_image->Name());
    if (!RNFileExists(filename)) 
      sprintf(filename, "%s/color_images/%s.jpg", input_image_directory, selected_image->Name());
    if (RNFileExists(filename)) {
      inset_image_pixels.ReadFile(filename);
      return;
    }
  }

  // Compute feature image
  if ((color_scheme == FEATURE_COLOR) && input_image_directory) {
    // Check if image feature file exists
    char filename[1024];
    sprintf(filename, "%s/clip_image_features/%s.npy", input_image_directory, selected_image->Name());
    if (RNFileExists(filename)) {
      // Read image features from file
      unsigned char *values = NULL;
      int data_type = 0, data_size = 0, fortran_order = 0, width = 0, height = 0, depth = 0;
      if (!ReadNumpyFile(filename, &data_type, &data_size,
        &fortran_order, &width, &height, &depth, &values)) {
        fprintf(stderr, "Unable to read npy file %s\n", filename);
        return;
      }

      // Determine scaling from surfel image to feature image
      double xscale = (double) height / (double) image_width;
      double yscale = (double) width  / (double) image_height;

      // Update inset image with pixel features
      for (int iy = 0; iy < image_height; iy++) {
        int features_iy = yscale * (image_height - iy - 1) + 0.5;
        if ((features_iy < 0) || (features_iy >= width)) continue;
        for (int ix = 0; ix < image_width; ix++) {
          int features_ix = xscale * ix + 0.5;
          if ((features_ix < 0) || (features_ix >= height)) continue;
          unsigned char *pixel_featuresp = values + 2 * depth * ((features_iy * height) + features_ix);
          half_float::half *pixel_features = (half_float::half *) pixel_featuresp;
          double r = pixel_features[0];
          double g = pixel_features[1];
          double b = pixel_features[2];
          r = r*r; g = g*g;  b= b*b;
          RNScalar sum = r + g + b;
          RNRgb color = (sum > 0) ? RNRgb(r/sum, g/sum, b/sum) : RNwhite_rgb;
          inset_image_pixels.SetPixelRGB(ix, iy, color);
        }
      }

      // Delete feature image
      delete [] values;
      return;
    }
  }

#if 0  
  // Compute affinity image
  if (((color_scheme == AFFINITY_COLOR) || (color_scheme == OVERLAY_COLOR)) && (query_features.NValues() > 0)) {
    // Read image features from file
    unsigned char *values = NULL;
    int data_type = 0, data_size = 0, fortran_order = 0, width = 0, height = 0, depth = 0;
    char filename[1024];
    sprintf(filename, "%s/clip_image_features/%s.npy", input_image_directory, selected_image->Name());
    if (!ReadNumpyFile(filename, &data_type, &data_size,
      &fortran_order, &width, &height, &depth, &values)) {
      fprintf(stderr, "Unable to read npy file %s\n", filename);
      return;
    }

    // Determine scaling from surfel image to feature image
    double xscale = (double) height / (double) image_width;
    double yscale = (double) width  / (double) image_height;

    // Update inset image with pixel affinities
    for (int iy = 0; iy < image_height; iy++) {
      int features_iy = yscale * (image_height - iy - 1) + 0.5;
      if ((features_iy < 0) || (features_iy >= width)) continue;
      for (int ix = 0; ix < image_width; ix++) {
        int features_ix = xscale * ix + 0.5;
        if ((features_ix < 0) || (features_ix >= height)) continue;
        unsigned char *pixel_featuresp = values + 2 * depth * ((features_iy * height) + features_ix);
        half_float::half *pixel_features = (half_float::half *) pixel_featuresp;
        RNScalar ssum = 0;
        for (int k = 0; k < depth; k++) ssum += pixel_features[k] * pixel_features[k];
        RNScalar l2norm = sqrt(ssum);
        RNScalar affinity = 0;
        for (int k = 0; k < depth; k++) affinity += query_features[k] * pixel_features[k] / l2norm;
        if ((color_scheme == AFFINITY_COLOR) || (affinity > value_range.Min())) {
          RNRgb color = NormalizedColor(affinity, color_scheme);
          inset_image_pixels.SetPixelRGB(ix, iy, color);
        }
      }
    }

    // Delete feature image
    delete [] values;
    return;
  }
#endif
  
  // Read color image by default
  if (input_image_directory) {
    // Read color image from file
    char filename[1024];
    sprintf(filename, "%s/color_images/%s.png", input_image_directory, selected_image->Name());
    if (!RNFileExists(filename)) 
      sprintf(filename, "%s/color_images/%s.jpg", input_image_directory, selected_image->Name());
    if (RNFileExists(filename)) {
      inset_image_pixels.ReadFile(filename);
      return;
    }
  }
}



////////////////////////////////////////////////////////////////////////
// Drawing functions
////////////////////////////////////////////////////////////////////////

static void
DrawMesh(int color_scheme)
{
  // Check display variables
  if (!show_vertices && !show_faces) return;

  // Check if color scheme is different
  if (color_scheme != vbo_color_scheme) {
    vbo_color_scheme = color_scheme;
    InvalidateVBO();
  }

  // Update VBOs
  UpdateVertexVBO(color_scheme);
  UpdateFaceVBO(color_scheme);

  // Check VBOs
  if (vbo_nvertices == 0) return;

  // Set opengl modes
  glEnable(GL_LIGHTING);
  glPointSize(2.0);

  // Enable vertex position buffer
  if (vbo_point_position_buffer > 0) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point_position_buffer);
    glVertexPointer(3, GL_FLOAT, 3 * sizeof(GLfloat), 0);
    glEnableClientState(GL_VERTEX_ARRAY);
  }

  // Enable vertex normal buffer
  if (vbo_point_normal_buffer > 0) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point_normal_buffer);
    glNormalPointer(GL_FLOAT,  3 * sizeof(GLfloat), 0);
    glEnableClientState(GL_NORMAL_ARRAY);
  }

  // Enable vertex color buffer
  if (vbo_point_color_buffer > 0) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point_color_buffer);
    glColorPointer(3, GL_UNSIGNED_BYTE, 3 * sizeof(GLubyte), 0);
    glEnableClientState(GL_COLOR_ARRAY);
  }

  // Draw vertices
  if (show_vertices) {
    glDrawArrays(GL_POINTS, 0, vbo_nvertices);
  }

  // Draw faces
  if (show_faces && (vbo_nfaces > 0) && (vbo_face_index_buffer > 0)) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_face_index_buffer);
    glDrawElements(GL_TRIANGLES, 3 * vbo_nfaces, GL_UNSIGNED_INT, 0);
  }

  // Disable client state
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  // Reset opengl modes
  glPointSize(1);
}



static void
DrawQueryString()
{
  // Check everything
  if (!show_query_string) return;
  if (query_string.empty()) return;

  // Draw text
  glDisable(GL_LIGHTING);
  RNLoadRgb(RNwhite_rgb - background);
  void *font = RN_GRFX_BITMAP_HELVETICA_18;
  viewer.DrawText(20, GLUTwindow_height - 40, query_string.c_str(), font);
}



static void
DrawCameras(int color_scheme)
{
  // Check display variables
  if (!show_cameras) return;

  // Draw cameras
  glDisable(GL_LIGHTING);
  for (int i = 0; i < surfels.NEntries(); i++) {
    R3SurfelScene *scene = surfels.Kth(i);
    for (int j = 0; j < scene->NImages(); j++) {
      R3SurfelImage *image = scene->Image(j);
      unsigned char r = (i+1) & 0xFF;
      unsigned char g = (j >> 8) & 0xFF;
      unsigned char b = j & 0xFF;
      if (color_scheme == PICK_COLOR) {
        RNLoadRgba(r, g, b, CAMERA_ALPHA);
        image->Draw();
      }
      else if (image == selected_image) {
        RNLoadRgb(0.5, 1, 1);
        glLineWidth(5);
        image->Draw();
        glLineWidth(1);
      }
      else {
        char filename[1024];
        sprintf(filename, "%s/clip_image_features/%s.npy", input_image_directory, image->Name());
        if (RNFileExists(filename)) {
          glLineWidth(2);
          RNLoadRgb(1, 1, 1);
          image->Draw();
          glLineWidth(1);
        }      
        else {
          RNLoadRgb(0, 1, 1);
          image->Draw();
        }
      }
    }
  }
}



static void 
DrawInsetImage(int color_scheme)
{
  // Get/check stuff
  if (!selected_image) return;
  if (selected_image->ImageWidth() <= 0) return;
  if (selected_image->ImageHeight() <= 0) return;
  if (!show_inset_image) return;
  if (inset_image_size <= 0) return;

  // Get/check color image
  UpdateInsetImage(color_scheme);
  int width = inset_image_pixels.Width();
  int height = inset_image_pixels.Height();
  if ((width <= 0) || (height <= 0)) return;
  const unsigned char *pixels = inset_image_pixels.Pixels();
  if (!pixels) return;

  // Determine image coordinates
  int w = viewer.Viewport().Width();
  int h = viewer.Viewport().Height();
  double x2 = w;
  double y2 = h;
  double aspect = (double) selected_image->ImageHeight() / (double) selected_image->ImageWidth();
  double x1 = x2 - inset_image_size * w;
  double y1 = y2 - inset_image_size * w * aspect;

  // Push ortho viewing matrices
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, w, 0, h, 0.1, 1);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Set viewport 
  glViewport(x1, y1, x2 - x1, y2 - y1);

  // Set image scale 
  double tx = 0;
  double ty = 0;
  GLfloat sx = (x2 - x1) / width;
  GLfloat sy = (y2 - y1) / height;
  glPixelZoom(sx, sy);

  // Set image translation
  // Can't send point outside viewport to glRasterPos, so add translation this way
  glRasterPos3d(0, 0, -0.5);
  glBitmap(0, 0, 0, 0, tx, ty, NULL);
  
  // Draw image
  glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

  // Reset everything
  glPixelZoom(1, 1);
  glViewport(0, 0, w, h);

  // Pop ortho viewing matrices
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}




static void
DrawCategoryNames()
{
  // Check everything
  if (!show_category_names) return;
  if (!category_names) return;
  if (mesh_segmentations.IsEmpty()) return;

  // Count vertices
  int vertex_count = 0;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes[m];
    vertex_count += mesh->NVertices();
  }

  // Compute step to limit how many category names to draw
  int max_draw_count = 1000;
  int step = vertex_count / max_draw_count + 1;

  // Draw category names for a sampling of points
  glDisable(GL_LIGHTING);
  void *font = RN_GRFX_BITMAP_HELVETICA_12;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes[m];
    if (m >= mesh_segmentations.NEntries()) break;
    RNVector *segmentation = mesh_segmentations[m];
    if (!segmentation) continue;
    for (int j = 0; j < mesh->NVertices(); j += step) {
      R3MeshVertex *vertex = mesh->Vertex(j);
      const R3Point& position = mesh->VertexPosition(vertex);
      int index = mesh->VertexValue(vertex) + 0.5;
      if (index >= segmentation->NValues()) continue;
      int category = (*segmentation)[index] + 0.5;
      if (category >= category_names->NEntries()) continue;
      RNLoadRgb(CategoryColor(category));
      R3DrawText(position, (*category_names)[category], font);
    }
  }
}



static void
DrawScene(void)
{
  // Check display variables
  if (!show_scene) return;
  if (!scene) return;
  
  // Draw scene
  glEnable(GL_LIGHTING);
  RNLoadRgb(1.0, 1.0, 1.0); 
  scene->Draw(R3_DEFAULT_DRAW_FLAGS);
}



static void
DrawAxes(void)
{
  // Check display variables
  if (!show_axes) return;
  
  // Draw axes 
  RNScalar d = 1;
  glLineWidth(3);
  R3BeginLine();
  RNLoadRgb(1, 0, 0);
  R3LoadPoint(R3zero_point + 0.5 * d * R3negx_vector);
  R3LoadPoint(R3zero_point + d * R3posx_vector);
  R3EndLine();
  R3BeginLine();
  RNLoadRgb(0, 1, 0);
  R3LoadPoint(R3zero_point + 0.5 * d * R3negy_vector);
  R3LoadPoint(R3zero_point + d * R3posy_vector);
  R3EndLine();
  R3BeginLine();
  RNLoadRgb(0, 0, 1);
  R3LoadPoint(R3zero_point + 0.5 * d * R3negz_vector);
  R3LoadPoint(R3zero_point + d * R3posz_vector);
  R3EndLine();
  glLineWidth(1);
}



static void
DrawSelectedPosition(void)
{
  // Check display variables
  if (!show_selected_position) return;
  if (selected_position == R3unknown_point) return;
  
  // Draw center point
  RNLoadRgb(0, 0, 0);
  RNScalar d = 0.025;
  R3Sphere(selected_position, d).Draw();
}



////////////////////////////////////////////////////////////////////////
// Selection with cursor
////////////////////////////////////////////////////////////////////////

static int
Pick(int x, int y, R3Point *picked_position = NULL,
  R3SurfelImage **picked_image = NULL, int pick_tolerance = 10)
{
  // Initialize pick results
  if (picked_position) *picked_position = R3unknown_point;
  if (picked_image) *picked_image = NULL;

  // Draw everything
  viewer.Load();
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPointSize(pick_tolerance);
  glLineWidth(pick_tolerance);
  EnableViewingExtent();
  DrawMesh(PICK_COLOR);
  DrawCameras(PICK_COLOR);
  DisableViewingExtent();
  glPointSize(1.0);
  glLineWidth(1.0);
  glFinish();

  // Read color buffer at cursor position
  unsigned char rgba[4];
  glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
  int r = rgba[0] & 0xFF;
  int g = rgba[1] & 0xFF;
  int b = rgba[2] & 0xFF;
  int a = rgba[3] & 0xFF;
  if ((r == 0) && (g == 0) && (b == 0)) return 0;

  // Check hit type
  if (a == CAMERA_ALPHA) {
    if (picked_image) {
      int scene_index = r-1;
      if ((scene_index >= 0) && (scene_index < surfels.NEntries())) {
        R3SurfelScene *scene = surfels.Kth(scene_index);
        int image_index = (g << 8) | b;
        if ((image_index >= 0) && (image_index < scene->NImages())) {
          *picked_image = scene->Image(image_index);
        }
      }
    }
  }

  // Return position
  if (picked_position) {
    // Find hit position
    GLfloat depth;
    GLdouble p[3];
    GLint viewport[4];
    GLdouble modelview_matrix[16];
    GLdouble projection_matrix[16];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
    glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    gluUnProject(x, y, depth, modelview_matrix, projection_matrix, viewport, &(p[0]), &(p[1]), &(p[2]));
    R3Point position(p[0], p[1], p[2]);
    *picked_position = position;
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Compute functions
////////////////////////////////////////////////////////////////////////

static RNVector
EncodeText(const std::string& str)
{
  // Initialize result
  RNVector features;
  int status = 0;

  // Make directory for temporary file
  std::string mkdir_cmd = "mkdir -p " + query_feature_directory;
  system(mkdir_cmd.c_str());

  /// Check if should use TCP connection
  if (use_tcp) {
    // Hard code tcp parameters for now
    int tcp_port = 1111;
    const char *tcp_server_hostname = "127.0.0.1"; // localhost
    RNInternetAddress tcp_addr = RNInternetAddressFromName(tcp_server_hostname);
    if (!tcp_addr) {
      RNAbort("Unable to get TCP address\n");
      return features;
    }

    // Open tcp connection (only first time)
    static RNTcp *tcp = NULL;
    if (!tcp) {
      tcp = new RNTcp(tcp_addr, tcp_port, FALSE);
      if (!tcp) {
        RNAbort("Unable to create TCP connection -- start server\n");
        return features;
      }
    }

    // Send query message 
    if (tcp->Write(query_string.c_str(), query_string.length()) <= 0) {
      RNFail("Failure during TCP send");
      return features;
    }

    // Wait for response
    char response[1024];
    int response_length = tcp->Read(response, 1023);
    if (response_length <= 0) {
      RNFail("Failure during TCP receive");
      return features;
    }

    // Check if response matches query
    response[response_length] = '\0';
    printf("--%s--\n", response);
    if (strcmp(response, query_string.c_str())) {
      RNFail("TCP response does not match query");
      return features;
    }

    // Set status -- success!
    status = 1;
  }
  else if (!query_feature_generator.empty()) {
    // Use standalone program to generate features (loads clip model each time)
    if (!query_feature_generator.empty()) {
      // Generate features and write them to a temporary file
      std::string cmd = query_feature_generator + " --out_dir " + query_feature_directory + " --text_prompt " + "\"" + query_string +"\"";
      system(cmd.c_str());
    }

    // Set status -- success!
    status = 1;
  }

  // Read back temporary file
  if (status) {
    std::string feat_file = query_feature_directory + "/" + query_string + ".npy";
    RNDenseMatrix *m = ReadFeaturesFile(feat_file.c_str());
    features.Reset(m->NColumns());
    for (int i = 0; i < m->NColumns(); i++) {
       features[i] = (*m)[0][i];
    }
    delete m;
  }

  // Return features
  return features;
}



static void
UpdateQueryFeatures(void)
{
  // Initialize query features
  query_features.Reset(0);
  
  // Check query string
  if (query_string.empty()) return;
  
  // Check input category list
  if (category_names && category_features) {
    // Find index of category matching query_string
    int category_index = -1;
    for (int i = 0; i < category_names->NEntries(); i++) {
      if (!strcmp((*category_names)[i], query_string.c_str())) {
        category_index = i;
        break;
      }
    }

    // Check if found matching category
    if (category_index >= 0) {
      // Copy features for the matching category
      query_features.Reset(category_features->NColumns());
      for (int i = 0; i < category_features->NColumns(); i++) {
        query_features[i] = (*category_features)[category_index][i];
      }

      // Remember category index
      selected_category_index = category_index;

      // Updated query features, can return
      return;
    }
  }

  // Generate features from scratch
  query_features = EncodeText(query_string);
}



static void
UpdateMeshAffinities(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate affinities
  for (int m = 0; m < point_features.NEntries(); m++) {
    if (mesh_affinities.NEntries() <= m) {
      RNDenseMatrix *features = point_features.Kth(m);
      RNVector *affinities = new RNVector(features->NRows());
      mesh_affinities.Insert(affinities);
    }
  }

  // Update affinities
  for (int m = 0; m < point_features.NEntries(); m++) {
    RNDenseMatrix *features = point_features.Kth(m);
    RNVector *affinities = mesh_affinities[m];
    if (!affinities) continue;
    
    // Check query features
    if (query_features.NValues() == features->NColumns()) {
      // Compute dot product (cosine similarity, since both matrices are normalized)
      for (int i = 0; i < features->NRows(); i++) {
        RNScalar value = 0;
        for (int j = 0; j < features->NColumns(); j++) {
          RNScalar mesh_value = (*features)[i][j];
          RNScalar query_value = query_features[j];
          value += mesh_value * query_value;
        }

        // Update affinity
        (*affinities)[i] = value;
      }
    }
    else {
      // Nonvalid query
      for (int i = 0; i < affinities->NValues(); i++) {
        affinities->SetValue(i, 0);
      }
    }
  }

  // Update max affinity
  max_affinity = 0;
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes.Kth(m);
    RNVector *affinities = mesh_affinities[m];
    if (!affinities) continue;
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      int index = mesh->VertexValue(vertex) + 0.5;
      RNScalar affinity = (*affinities)[index];
      if (affinity > max_affinity) {
        center = mesh->VertexPosition(mesh->Vertex(i));
        selected_position = center;
        max_affinity = affinity;
      }
    }
  }

  // Invalidate VBO
  InvalidateVBO();

  // Print statistics
  if (print_debug) {
    printf("Updated mesh affinities ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }
}



static void
UpdateMeshSegmentations(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Check mesh and category features
  if (point_features.IsEmpty()) return;
  if (!category_features) return;

  // Allocate segmentations
  for (int m = 0; m < point_features.NEntries(); m++) {
    if (mesh_segmentations.NEntries() <= m) {
      RNDenseMatrix *features = point_features.Kth(m);
      RNVector *segmentation = new RNVector(features->NRows());
      mesh_segmentations.Insert(segmentation);
    }
  }

  // Update segmentations
  for (int m = 0; m < mesh_segmentations.NEntries(); m++) {
    RNDenseMatrix *features = point_features.Kth(m);
    RNVector *segmentation = mesh_segmentations[m];
    if (!segmentation) continue;
    for (int i = 0; i < features->NRows(); i++) {
      RNScalar best_affinity = 0;
      for (int j = 0; j < category_features->NRows(); j++) {
        RNScalar affinity = 0;
        for (int k = 0; k < category_features->NColumns(); k++) {
          RNScalar mesh_value = (*features)[i][k];
          RNScalar category_value = (*category_features)[j][k];
          affinity += mesh_value * category_value;
        }
        if (affinity > best_affinity) {
          best_affinity = affinity;
          (*segmentation)[i] = j;
        }
      }
    }
  }

  // Invalidate VBO
  InvalidateVBO();

  // Print statistics
  if (print_debug) {
    printf("Updated mesh segmentations ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }
}



static void
SelectCategory(int category_index)
{
  // Update query string
  query_string.clear();
  if (category_names) {
    if ((category_index >= 0) && (category_index < category_names->NEntries())) {
      query_string = std::string((*category_names)[category_index]);
    }
  }

  // Update query features
  query_features.Reset(0);
  if (category_features) {
    if ((category_index >= 0) && (category_index < category_features->NRows())) {
      query_features.Reset(category_features->NColumns());
      for (int i = 0; i < category_features->NColumns(); i++) {
        query_features[i] = (*category_features)[category_index][i];
      }
    }
  }

  // Remember selected category
  selected_category_index = category_index;
}



////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static void
ResetViewer(void)
{
  // Get bounding box
  scene_extent = R3null_box;
  for (int i = 0; i < meshes.NEntries(); i++) {
    R3Mesh *mesh = meshes.Kth(i);
    scene_extent.Union(mesh->BBox());
  }

  // Initialize viewing center
  center = scene_extent.Centroid();

  // Initialize viewer
  RNLength r = scene_extent.DiagonalRadius();
  if (r < 10) r = 10;
  R3Point eye = center - R3negz_vector * (2.5 * r);
  R3Camera camera(eye, R3negz_vector, R3posy_vector, 0.4, 0.4, 0.01 * r, 100.0 * r);
  R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
  viewer.SetViewport(viewport);
  viewer.SetCamera(camera);
}



////////////////////////////////////////////////////////////////////////
// GLUT user interface functions
////////////////////////////////////////////////////////////////////////

void GLUTStop(void)
{
  // Close surfels files
  for (int i = 0; i < surfels.NEntries(); i++) {
    R3SurfelScene *scene = surfels.Kth(i);
    if (!CloseSurfelsFiles(scene)) exit(-1);
  }

  // Delete VBO buffers
  if (vbo_point_position_buffer > 0) glDeleteBuffers(1, &vbo_point_position_buffer);
  if (vbo_point_normal_buffer > 0) glDeleteBuffers(1, &vbo_point_normal_buffer);
  if (vbo_point_color_buffer > 0) glDeleteBuffers(1, &vbo_point_color_buffer);
  if (vbo_face_index_buffer > 0) glDeleteBuffers(1, &vbo_face_index_buffer);

  // Destroy window 
  glutDestroyWindow(GLUTwindow);

  // Exit
  exit(0);
}



void GLUTRedraw(void)
{
  // Set viewing transformation
  viewer.Camera().Load();

  // Clear window 
  // glClearColor(0.0, 0.0, 0.0, 1.0);
  glClearColor(background.R(), background.G(), background.B(), 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw everything to be clipped by viewing extent
  DrawViewingExtent();
  EnableViewingExtent();
  DrawMesh(color_scheme);
  DrawCameras(color_scheme);
  DrawCategoryNames();
  DrawScene();
  DisableViewingExtent();

  // Draw everything not being clipped by viewing extent
  DrawAxes();
  DrawQueryString();
  DrawSelectedPosition();
  DrawInsetImage(color_scheme);

  // Capture screenshot image 
  if (screenshot_image_name) {
    if (print_verbose) printf("Creating image %s\n", screenshot_image_name);
    R2Image image(GLUTwindow_width, GLUTwindow_height, 3);
    image.Capture();
    image.Write(screenshot_image_name);
    screenshot_image_name = NULL;
  }

  // Swap buffers 
  glutSwapBuffers();
}    



void GLUTResize(int w, int h)
{
  // Resize window
  glViewport(0, 0, w, h);

  // Resize viewer viewport
  viewer.ResizeViewport(0, 0, w, h);

  // Remember window size 
  GLUTwindow_width = w;
  GLUTwindow_height = h;

  // Redraw
  glutPostRedisplay();
}



void GLUTMotion(int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Compute mouse movement
  int dx = x - GLUTmouse[0];
  int dy = y - GLUTmouse[1];
  
  // Update mouse drag
  GLUTmouse_drag += dx*dx + dy*dy;

  // World in hand navigation 
  if (GLUTbutton[0]) viewer.RotateWorld(1.0, center, x, y, dx, dy);
  else if (GLUTbutton[1]) viewer.ScaleWorld(1.0, center, x, y, dx, dy);
  else if (GLUTbutton[2]) viewer.TranslateWorld(1.0, center, x, y, dx, dy);
  if (GLUTbutton[0] || GLUTbutton[1] || GLUTbutton[2]) glutPostRedisplay();

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;
}



void GLUTMouse(int button, int state, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Mouse is going down
  if (state == GLUT_DOWN) {
    // Reset mouse drag
    GLUTmouse_drag = 0;

    // Process thumbwheel
    if (button == 3) viewer.ScaleWorld(center, 0.9);
    else if (button == 4) viewer.ScaleWorld(center, 1.1);
  }
  else if (button == 0) {
    // Check for double click  
    static RNBoolean double_click = FALSE;
    static RNTime last_mouse_up_time;
    double_click = (!double_click) && (last_mouse_up_time.Elapsed() < 0.4);
    last_mouse_up_time.Read();

    // Check for click (rather than drag)
    if (GLUTmouse_drag < 100) {
      // Select image or surface
      R3SurfelImage *picked_image = NULL;
      selected_position = R3unknown_point;
      if (Pick(x, y, &selected_position, &picked_image)) {
        if (picked_image) { printf("%s\n", picked_image->Name()); selected_image = picked_image; }
        else if (!surfels.IsEmpty()) selected_image = surfels[0]->FindImageByBestView(selected_position, R3zero_vector);
        center = selected_position;
      }
    }
  }

  // Remember button state 
  int b = (button == GLUT_LEFT_BUTTON) ? 0 : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
  GLUTbutton[b] = (state == GLUT_DOWN) ? 1 : 0;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

   // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Redraw
  glutPostRedisplay();
}



void GLUTSpecial(int key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event
  switch (key) {
  case GLUT_KEY_F1: {
    // Dump screen shot to file iX.jpg
    static char buffer[64];
    static int image_count = 1;
    sprintf(buffer, "i_%s_%d_%d.jpg", query_string.c_str(), color_scheme, image_count++);
    screenshot_image_name = buffer;
    break; }

  case GLUT_KEY_F2: {
    // Print camera
    const R3Camera& camera = viewer.Camera();
    printf("%g %g %g  %g %g %g  %g %g %g  %g %g  1\n",
           camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
           camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
           camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
           camera.XFOV(), camera.YFOV());
    break; }

  case GLUT_KEY_F7:
    // Jump to viewpoint of selected image
    if (selected_image) {
      R3Camera camera(selected_image->Viewpoint(), selected_image->Towards(), selected_image->Up(),
         selected_image->XFOV(), selected_image->YFOV(), viewer.Camera().Near(), viewer.Camera().Far());
      viewer.SetCamera(camera);
    }
    break;

  case GLUT_KEY_HOME:
  case GLUT_KEY_END:
  case GLUT_KEY_PAGE_DOWN:
  case GLUT_KEY_PAGE_UP: 
    if (category_features) {
      int category_index = selected_category_index;
      if (key == GLUT_KEY_PAGE_DOWN) category_index--;
      else if (key == GLUT_KEY_PAGE_UP) category_index++;
      else if (key == GLUT_KEY_HOME) category_index = 0;
      else if (key == GLUT_KEY_END) category_index = INT_MAX;
      if (category_index < 0) category_index = 0;
      if (category_index >= category_features->NRows()) category_index = category_features->NRows()-1;
      SelectCategory(category_index);
      UpdateMeshAffinities();
      InvalidateVBO();
    }
    break; 

  case GLUT_KEY_RIGHT:
  case GLUT_KEY_LEFT:
  case GLUT_KEY_DOWN:
  case GLUT_KEY_UP: {
    RNScalar min_diameter = 0.01;
    RNScalar min_value = value_range.Min();
    RNScalar max_value = value_range.Max();
    RNScalar diameter = value_range.Diameter();
    RNScalar scale = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) ? 0.25 : 0.1;
    if (key == GLUT_KEY_LEFT) { min_value += 0.1 * diameter; max_value -= scale * diameter; }
    else if (key == GLUT_KEY_RIGHT) { min_value -= scale * diameter; max_value += scale * diameter; }
    else if (key == GLUT_KEY_DOWN) { min_value -= 0.1 * scale; max_value -= 0.1 * scale; }
    else if (key == GLUT_KEY_UP) { min_value += 0.1 * scale; max_value += 0.1 * scale; }
    if (min_value < 0) min_value = 0;
    if (min_value > max_affinity - min_diameter) min_value = max_affinity - min_diameter;
    if (max_value > max_affinity) max_value = max_affinity;
    if (max_value < min_value + min_diameter) max_value = min_value + min_diameter;
    value_range.Reset(min_value, max_value);
    InvalidateVBO();
    break; }
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}



void GLUTKeyboard(unsigned char key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Check if alt key is down
  if (glutGetModifiers() & GLUT_ACTIVE_ALT) {
    // Process alt key events
    switch (key) {
    case 'A':
    case 'a':
      show_axes = !show_axes;
      break;

    case 'C':
    case 'c':
      if (color_scheme == OVERLAY_COLOR) color_scheme = AFFINITY_COLOR;
      else if (color_scheme == AFFINITY_COLOR) color_scheme = FEATURE_COLOR;
      else if (color_scheme == FEATURE_COLOR) color_scheme = SEGMENTATION_COLOR;
      else if (color_scheme == SEGMENTATION_COLOR) color_scheme = RGB_COLOR;
      else color_scheme = OVERLAY_COLOR;
      InvalidateVBO();
      break;

    case 'F':
    case 'f':
      show_faces = !show_faces;
      break;

    case 'I':
    case 'i':
      show_inset_image = !show_inset_image;
      break;

    case 'P':
    case 'p':
      show_cameras = !show_cameras;
      break;

    case 'Q':
    case 'q':
      show_query_string = !show_query_string;
      break;

    case 'N':
    case 'n':
      show_category_names = !show_category_names;
      break;

    case 'T':
    case 't':
      show_scene = !show_scene;
      break;

    case 'V':
    case 'v':
      show_vertices = !show_vertices;
      break;
      
    case 'W':
    case 'w':
      show_weak_affinities = !show_weak_affinities;
      InvalidateVBO();
      break;

    case 'Y':
    case 'y':
      show_selected_position = !show_selected_position;
      break;
    }
  }
  else {
    switch (key) {
    case '=':
      // Grow inset image
      inset_image_size *= 1.25;
      break;

    case '-':
      // Shrink inset image
      inset_image_size *= 0.8;
      break;

    case '+':
      // Grow viewing extent
      if (viewing_extent.IsEmpty()) viewing_extent = scene_extent;
      if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) viewing_extent[RN_LO][RN_Z] += 0.01 * scene_extent.ZLength();
      else viewing_extent[RN_HI][RN_Z] += 0.01 * scene_extent.ZLength();
      if (R3Contains(viewing_extent, scene_extent)) viewing_extent = R3null_box;
      break;

    case '_':
      // Shrink viewing extent
      if (viewing_extent.IsEmpty()) viewing_extent = scene_extent;
      if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) viewing_extent[RN_LO][RN_Z] -= 0.01 * scene_extent.ZLength();
      else viewing_extent[RN_HI][RN_Z] -= 0.01 * scene_extent.ZLength();
      if (R3Contains(viewing_extent, scene_extent)) viewing_extent = R3null_box;
      break;

    case 8: // backspace
    case 127: // delete
      // Remove one char from query string
      if (query_string.size() > 0) {
        query_string.erase(query_string.end()-1);
      }
      break;

    case 13: // Enter
      if (query_string.size() > 0) {
        UpdateQueryFeatures();
        UpdateMeshAffinities();
      }
      break;

    case 17: // ctrl-Q
      GLUTStop();
      break;

    case 18: // ctrl-R
      ResetViewer();
      break;

    case 27: // ESCAPE
      // Reset everything
      value_range = default_value_range;
      query_string.clear();
      UpdateQueryFeatures();
      UpdateMeshAffinities();
      InvalidateVBO();
      break;

    default:
      // Add one char to query string
      if ((key >= 32) && (key <= 126)) {
        query_string.append(1, key);
      }
    }
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = GLUTwindow_height - y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();  
}




void GLUTInterface(void)
{
  // Open window
  int argc = 0;
  char *argv[1];
  argv[argc++] = RNStrdup("conf2texture");
  glutInit(&argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA);
  GLUTwindow = glutCreateWindow("OpenScene Viewer");
  
  // Initialize grfx (after create context because calls glewInit)
  RNInitGrfx();

  // Initialize graphics modes  
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  // Initialize lighting
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  static GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
  glEnable(GL_NORMALIZE);

  // Define headlight
  // static GLfloat light0_diffuse[] = { 0.5, 0.5, 0.5, 1.0 };
  static GLfloat light0_diffuse[] = { 1, 1, 1, 1.0 };
  static GLfloat light0_position[] = { 0.0, 0.0, 1.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  glEnable(GL_LIGHT0);

  // Initialize color settings
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

  // Initialize GLUT callback functions 
  glutDisplayFunc(GLUTRedraw);
  glutReshapeFunc(GLUTResize);
  glutKeyboardFunc(GLUTKeyboard);
  glutSpecialFunc(GLUTSpecial);
  glutMouseFunc(GLUTMouse);
  glutMotionFunc(GLUTMotion);

  // Reset viewer
  ResetViewer();

  // Run main loop -- never returns 
  glutMainLoop();
}


 
////////////////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-debug")) print_debug = 1;
      else if (!strcmp(*argv, "-tcp")) use_tcp = 1;
      else if (!strcmp(*argv, "-one_feature_vector_per_object")) one_feature_vector_per_object = TRUE;
      else if (!strcmp(*argv, "-scene")) { argc--; argv++; input_scene_filename = *argv; }
      else if (!strcmp(*argv, "-category_names")) { argc--; argv++; input_category_names_filename = *argv; }
      else if (!strcmp(*argv, "-category_colors")) { argc--; argv++; input_category_colors_filename = *argv; }
      else if (!strcmp(*argv, "-category_features")) { argc--; argv++; input_category_features_filename = *argv; }
      else if (!strcmp(*argv, "-image_directory")) { argc--; argv++; input_image_directory = *argv; }
      else if (!strcmp(*argv, "-window")) { 
        argv++; argc--; GLUTwindow_width = atoi(*argv); 
        argv++; argc--; GLUTwindow_height = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-value_range")) { 
        argv++; argc--; default_value_range.SetMin(atof(*argv)); 
        argv++; argc--; default_value_range.SetMax(atof(*argv)); 
      }
      else if (!strcmp(*argv, "-background")) { 
        argv++; argc--; background[0] = atof(*argv); 
        argv++; argc--; background[1] = atof(*argv); 
        argv++; argc--; background[2] = atof(*argv);
      }
      else {
        RNFail("Invalid program argument: %s", *argv);
        exit(1);
      }
      argv++; argc--;
    }
    else {
      if (strstr(*argv, ".ply")) input_mesh_filenames.Insert(*argv);
      else if (strstr(*argv, ".ssa")) input_ssa_filenames.Insert(*argv);
      else if (strstr(*argv, ".ssb")) input_ssb_filenames.Insert(*argv);
      else if (strstr(*argv, ".npy")) input_point_features_filenames.Insert(*argv);
      else if (strstr(*argv, ".conf")) input_configuration_filenames.Insert(*argv);
      else { RNFail("Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check filenames
  if ((input_ssa_filenames.IsEmpty() || input_ssb_filenames.IsEmpty()) &&
      (input_mesh_filenames.IsEmpty() || input_point_features_filenames.IsEmpty())) {
    RNFail("Usage: osview inputmesh inputfeatures [options]\n");
    return 0;
  }

  // Set display variables
  if (input_mesh_filenames.IsEmpty()) show_vertices = 1;

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////

int
main(int argc, char **argv)
{
  // Initialize packages
  if (!R3InitGraphics()) exit(-1);

  // Check number of arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read meshes
  for (int i = 0; i < input_mesh_filenames.NEntries(); i++) {
    R3Mesh *mesh = ReadMeshFile(input_mesh_filenames[i]);
    if (!mesh) exit(-1);
    meshes.Insert(mesh);
  }

  // Open surfels files
  for (int i = 0; i < input_ssa_filenames.NEntries(); i++) {
    if (i >= input_ssb_filenames.NEntries()) break;
    R3SurfelScene *scene = OpenSurfelsFiles(input_ssa_filenames[i], input_ssb_filenames[i]);
    if (!scene) exit(-1);
    surfels.Insert(scene);
  }

  // Create meshes from surfels
  for (int i = 0; i < surfels.NEntries(); i++) {
    R3SurfelScene *scene = surfels.Kth(i);
    if (scene->NSurfels() == 0) continue;
    R3Mesh *mesh = CreateMeshFromSurfels(scene);
    if (!mesh) continue;
    meshes.Insert(mesh);
  }

  // Read configuration files
  for (int i = 0; i < input_configuration_filenames.NEntries(); i++) {
    R3SurfelScene *scene = ReadConfigurationFile(input_configuration_filenames[i]);
    if (!scene) exit(-1);
    surfels.Insert(scene);
  }

  // Read point features
  for (int i = 0; i < input_point_features_filenames.NEntries(); i++) {
    RNDenseMatrix *features = ReadFeaturesFile(input_point_features_filenames[i]);
    if (!features) exit(-1);
    point_features.Insert(features);
  }

  // Read category features
  if (input_category_features_filename) {
    category_features = ReadFeaturesFile(input_category_features_filename);
    if (!category_features) exit(-1);
  }
  
  // Read category names
  if (input_category_names_filename) {
    category_names = ReadCategoryNamesFile(input_category_names_filename);
    if (!category_names) exit(-1);
  }
  
  // Read category colors
  if (input_category_colors_filename) {
    category_colors = ReadCategoryColorsFile(input_category_colors_filename);
    if (!category_colors) exit(-1);
  }
  
  // Read scene
  if (input_scene_filename) {
    scene = ReadSceneFile(input_scene_filename);
    if (!scene) exit(-1);
  }

  // Compute affinities
  UpdateMeshAffinities();

  // Compute affinities
  UpdateMeshSegmentations();

  // Begin viewing interface -- never returns
  GLUTInterface();

  // Return success 
  return 0;
}



