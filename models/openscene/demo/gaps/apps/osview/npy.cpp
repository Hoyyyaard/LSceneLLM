#include "RNBasics/RNBasics.h"
#include "npy.h"


static int
DataTypeSize(int data_type, int data_size)
{
  switch (data_type) {
  case 'U': return 4 * data_size;
  default: return data_size;
  }
}



int
ReadNumpyFile(const char *filename,
  int *returned_data_type, int *returned_data_size, int *returned_fortran_order,
  int *returned_width, int *returned_height, int *returned_depth,
  unsigned char **returned_array)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open npy file %s\n", filename);
    return 0;
  }
  
  // Read magic string
  unsigned char magic[6];
  if (fread(magic, sizeof(unsigned char), 6, fp) != (unsigned int) 6) {
    fprintf(stderr, "Unable to read npy file %s\n", filename);
    fclose(fp);
    return 0;
  }

  // Check magic string
  if ((magic[0] != 0x93) || (magic[1] != 'N') || (magic[2] != 'U') ||
      (magic[3] != 'M')  || (magic[4] != 'P') || (magic[5] != 'Y')) {
      fprintf(stderr, "Unrecognized format in npy file %s\n", filename);
    fclose(fp);
    return 0;
  }

  // Read version info
  unsigned char version[2];
  if (fread(version, sizeof(unsigned char), 2, fp) != (unsigned int) 2) {
    fprintf(stderr, "Unable to read version in npy file %s\n", filename);
    fclose(fp);
    return 0;
  }
  
  // Read header length
  unsigned short int header_length;
  if (fread(&header_length, sizeof(unsigned short), 1, fp) != (unsigned int) 1) {
    fprintf(stderr, "Unable to read header length in npy file %s\n", filename);
    fclose(fp);
    return 0;
  }

  // Check header length
  if (header_length <= 0) {
    fprintf(stderr, "Invalid header length in npy file %s\n", filename);
    fclose(fp);
    return 0;
  }
  
  // Read header
  char *header = new char [ header_length ];
  if (fread(header, sizeof(char), header_length, fp) != (unsigned int) header_length) {
    fprintf(stderr, "Unable to read header in npy file %s\n", filename);
    delete [] header;
    fclose(fp);
    return 0;
  }

  // Extract data type and size
  int data_type = 0;
  int data_size = 0;
  char *start = strstr(header, "'descr'");
  if (start) {
    start = strchr(start, '<');
    if (start) {
      start++;
      while (*start == ' ') start++;
      data_type = *start;
      start++;
      char *end = strchr(start, '\'');
      if (end && (start < end)) {
        *end = '\0';
        data_size = atoi(start);
        *end = '\'';
      }
    }
  }
  
  
  // Extract fortrain order
  int fortran_order = 0;
  start = strstr(header, "'fortran_order'");
  if (start) {
    start = strchr(start, ':');
    if (start) {
      start++;
      while (*start == ' ') start++;
      char *end = strchr(start, ',');
      if (end && (start < end)) {
        *end = '\0';
        if (!strcmp(start, "True")) fortran_order = 1;
        if (!strcmp(start, "true")) fortran_order = 1;
        *end = ',';
      }
    }
  }
    
  // Extract width
  int width = 1;
  start = strstr(header, "'shape'");
  if (start) {
    start = strchr(start, '(');
    if (start) {
      start++;
      while (*start == ' ') start++;
      char *end = strchr(start, ',');
      if (end && (start < end)) {
        *end = '\0';
        width = atoi(start);
        *end = ',';
      }
    }
  }
  
  // Extract height
  int height = 1;
  start = strstr(header, "'shape'");
  if (start) {
    start = strchr(start, ',');
    if (start) {
      start++;
      while (*start == ' ') start++;
      char *end1 = strchr(start, ',');
      char *end2 = strchr(start, ')');
      char *end = (end1 < end2) ? end1 : end2;
      if (end && (start < end)) {
        *end = '\0';
        height = atoi(start);
        *end = ')';
      }
    }
  }

  // Extract depth
  int depth = 1;
  start = strstr(header, "'shape'");
  if (start) {
    start = strchr(start, ',');
    if (start) {
      start++;
      while (*start == ' ') start++;
      char *end1 = strchr(start, ',');
      char *end2 = strchr(start, ')');
      char *end = (end1 < end2) ? end1 : end2;
      if (end && (start < end)) {
        start = end + 1;
        while (*start == ' ') start++;
        char *end1 = strchr(start, ',');
        char *end2 = strchr(start, ')');
        char *end = (end1 < end2) ? end1 : end2;
        if (end && (start < end)) {
          *end = '\0';
          depth = atoi(start);
          *end = ')';
        }
      }
    }
  }

  // Delete header
  delete [] header;

  // Read array
  unsigned char *array = NULL;
  if (returned_array) {
    unsigned int nbytes = width * height * depth * DataTypeSize(data_type, data_size);
    array = new unsigned char [ nbytes ];
    if (fread(array, sizeof(unsigned char), nbytes, fp) != (unsigned int) nbytes) {
      fprintf(stderr, "Unable to read array in npy file %s\n", filename);
      delete [] array;
      fclose(fp);
      return 0;
    }
  }

  // Close file
  fclose(fp);

  // Return width and height
  if (returned_data_type) *returned_data_type = data_type;
  if (returned_data_size) *returned_data_size = data_size;
  if (returned_fortran_order) *returned_fortran_order = fortran_order;
  if (returned_width) *returned_width = width;
  if (returned_height) *returned_height = height;
  if (returned_height) *returned_depth = depth;
  if (returned_array) *returned_array = array;
  
  // Return success
  return 1;
}




