#include "JPEG2000Codec.h"
#include "opj_config.h"
#include "openjpeg.h"
#include <string>
#include <vector>
#include <sstream>

using namespace std;

void error_callback(const char *msg, void *client_data)
{
}

/* Warning callback expecting a FILE* client object. */
void warning_callback(const char *msg, void *client_data)
{
}

/* Debug callback expecting no client object. */
void info_callback(const char *msg, void *client_data)
{
}

typedef struct
{
  OPJ_UINT8* data;
  OPJ_SIZE_T size;
  OPJ_SIZE_T offset;
} opj_memory_stream;

//This will read from p_buffer to the stream.
static OPJ_SIZE_T opj_memory_stream_read(void * p_buffer, OPJ_SIZE_T p_nb_bytes, void * p_user_data)
{
  opj_memory_stream* l_memory_stream = (opj_memory_stream*)p_user_data;//Our data.
  OPJ_SIZE_T dataEndOffset = l_memory_stream->size - 1;
  OPJ_SIZE_T l_nb_bytes_read = p_nb_bytes;//Amount to move to buffer.  

  //Check if the current offset is outside our data buffer.  
  if (l_memory_stream->offset >= dataEndOffset) {
    return (OPJ_SIZE_T)-1;
  }
  
  //Check if we are reading more than we have.
  if (p_nb_bytes > (dataEndOffset - l_memory_stream->offset)) {
    l_nb_bytes_read = dataEndOffset - l_memory_stream->offset;  //Read all we have.
  }

  //Copy the data to the internal buffer.
  memcpy(p_buffer, &(l_memory_stream->data[l_memory_stream->offset]), l_nb_bytes_read);
  l_memory_stream->offset += l_nb_bytes_read;//Update the pointer to the new location.
  return l_nb_bytes_read;
}

//This will write from the stream to p_buffer.
static OPJ_SIZE_T opj_memory_stream_write(void * p_buffer, OPJ_SIZE_T p_nb_bytes, void * p_user_data)
{
  opj_memory_stream* l_memory_stream = (opj_memory_stream*)p_user_data; //Our data.
  OPJ_SIZE_T dataEndOffset = l_memory_stream->size - 1;
  OPJ_SIZE_T l_nb_bytes_write = p_nb_bytes; //Amount to move to buffer.
 
  //Check if the current offset is outside our data buffer.
  if (l_memory_stream->offset >= dataEndOffset) {
    return (OPJ_SIZE_T)-1;
  }
  
  //Check if we are write more than we have space for.
  if (p_nb_bytes > (dataEndOffset - l_memory_stream->offset)) {
    l_nb_bytes_write = dataEndOffset - l_memory_stream->offset; //Write the remaining space.
  }
  
  //Copy the data from the internal buffer.
  memcpy(&(l_memory_stream->data[l_memory_stream->offset]), p_buffer, l_nb_bytes_write);
  l_memory_stream->offset += l_nb_bytes_write;//Update the pointer to the new location.
  return l_nb_bytes_write;
}

//Moves the current offset forward, but never more than size.
static OPJ_OFF_T opj_memory_stream_skip(OPJ_OFF_T p_nb_bytes, void * p_user_data)
{
  opj_memory_stream* l_memory_stream = (opj_memory_stream*)p_user_data;
  OPJ_SIZE_T dataEndOffset = l_memory_stream->size - 1;
  OPJ_SIZE_T l_nb_bytes;

  if (p_nb_bytes < 0) {
    return -1;
  }
  l_nb_bytes = (OPJ_SIZE_T)p_nb_bytes;
  
  // Do not allow moving past the end.
  if (l_nb_bytes > dataEndOffset - l_memory_stream->offset) {
    l_nb_bytes = dataEndOffset - l_memory_stream->offset;
  }

  //Make the jump.
  l_memory_stream->offset += l_nb_bytes;
  return l_nb_bytes;
}

//Sets the offset to anywhere in the stream.
static OPJ_BOOL opj_memory_stream_seek(OPJ_OFF_T p_nb_bytes, void * p_user_data)
{
  opj_memory_stream* l_memory_stream = (opj_memory_stream*)p_user_data;

  if (p_nb_bytes < 0) {
    return OPJ_FALSE; //Not before the buffer.
  } 

  if (p_nb_bytes > (OPJ_OFF_T)(l_memory_stream->size - 1)) {
    return OPJ_FALSE; //Not after the buffer.
  }

  l_memory_stream->offset = (OPJ_SIZE_T)p_nb_bytes; //Move to new position.
  return OPJ_TRUE;
}

static void opj_memory_stream_do_nothing(void * p_user_data)
{
  OPJ_ARG_NOT_USED(p_user_data);
}

//Create a stream to use memory as the input or output.
opj_stream_t* opj_stream_create_default_memory_stream(opj_memory_stream* p_memoryStream, OPJ_BOOL is_read_stream)
{
  opj_stream_t* l_stream;

  if (!(l_stream = opj_stream_default_create(is_read_stream))) {
    return (NULL);
  }

  //Set how to work with the frame buffer.
  if (is_read_stream) {
    opj_stream_set_read_function(l_stream, opj_memory_stream_read);
  }
  else {
    opj_stream_set_write_function(l_stream, opj_memory_stream_write);
  }
  opj_stream_set_seek_function(l_stream, opj_memory_stream_seek);
  opj_stream_set_skip_function(l_stream, opj_memory_stream_skip);
  opj_stream_set_user_data(l_stream, p_memoryStream, opj_memory_stream_do_nothing);
  opj_stream_set_user_data_length(l_stream, p_memoryStream->size);
  return l_stream;
}


JPEG2000Codec::JPEG2000Codec()
{
}

JPEG2000Codec::~JPEG2000Codec() {
}

void JPEG2000Codec::decode(unsigned char* buf, const unsigned int& inSize, const unsigned int& outSize)
{
  //Kind of Openjpeg Stuff (shound be)
  opj_memory_stream decodeStream;
  // OpenJPEG stuff.
  opj_image_t			*decompImage = NULL;
  opj_stream_t		*l_stream = NULL;

  //Set up the input buffer as a stream
  decodeStream.data = (OPJ_UINT8 *)buf;
  decodeStream.size = inSize;
  decodeStream.offset = 0;
    //Open the memory as a stream.
  l_stream = opj_stream_create_default_memory_stream(&decodeStream, OPJ_TRUE);

  opj_dparameters_t decodeParameters;
  opj_set_default_decoder_parameters(&decodeParameters);
  opj_codec_t* decoder = opj_create_decompress(OPJ_CODEC_FORMAT::OPJ_CODEC_J2K);//Bad openjpeg, can't be reused!
  //Catch events using our callbacks and give a local context.
  opj_set_info_handler(decoder, info_callback, NULL);
  opj_set_warning_handler(decoder, warning_callback, NULL);
  opj_set_error_handler(decoder, error_callback, NULL);

  //Setup the decoder decoding parameters using user parameters.
  opj_setup_decoder(decoder, &decodeParameters);
  opj_codec_set_threads(decoder, 4);

  // Read the main header of the codestream, if necessary the JP2 boxes, and create decompImage.
  OPJ_BOOL headerSucces = opj_read_header(l_stream, decoder, &decompImage);
  OPJ_BOOL decodeSucces = opj_decode(decoder, l_stream, decompImage);
  OPJ_BOOL decompressSuccess = opj_end_decompress(decoder, l_stream);
  
  //Done with the input stream and the decoder.
  opj_stream_destroy(l_stream);
  opj_destroy_codec(decoder);

  //Get the total uncompressed length.
  int prec_jpc = decompImage->comps[0].prec; //Need it at the end
  int bytes_jpc= (prec_jpc + 7) / 8;// Bytes or words.
  int stream_len = (decompImage->comps[0].w * decompImage->comps[0].h);

  OPJ_INT32 ** compPointers = new OPJ_INT32*[decompImage->numcomps];
  for (int cmp = 0; cmp < decompImage->numcomps; cmp++) {
    compPointers[cmp] = decompImage->comps[cmp].data;
  }
  std::fill(buf, buf + outSize, 0);
  for (int index = 0; index < stream_len; index++) {
    for (int cmp = 0; cmp < decompImage->numcomps; cmp++) {
      for (int byteCnt = 0; byteCnt < bytes_jpc; byteCnt++)
      {
        *(buf)++ |= (unsigned char)((*compPointers[cmp] >> (8 * byteCnt)) & 0xFF);
      }
      *compPointers[cmp]++;
    }
  }
  opj_image_destroy(decompImage);
  delete[] compPointers;
}

void JPEG2000Codec::encode(char* data, unsigned int& size, const unsigned int& tileSize, const unsigned int& rate, const unsigned int& nrComponents, const pathology::DataType& dataType, const pathology::ColorType& colorSpace) const
{

  int depth = 8;
  if (dataType == pathology::DataType::Float || dataType == pathology::DataType::InvalidDataType) {
    return;
  }
  else if (dataType == pathology::DataType::UInt16) {
    depth = 16;
  }
  else if (dataType == pathology::DataType::UInt32 && colorSpace != pathology::ColorType::ARGB) {
    depth = 32;
  }
  
  opj_cparameters_t	encodeParameters;	// compression parameters.
  encodeParameters.tcp_mct = nrComponents > 1 ? 1 : 0; //Decide if MCT should be used.
  opj_set_default_encoder_parameters(&encodeParameters);
  if (rate < 100)
  {
    encodeParameters.tcp_rates[0] = 100.0f / rate;
    encodeParameters.irreversible = 1;//ICT
  }
  else {
    encodeParameters.tcp_rates[0] = 0;
  }
  encodeParameters.tcp_numlayers = 1;
  encodeParameters.cp_disto_alloc = 1;

  //Set the image components parameters.
  opj_image_cmptparm_t*componentParameters = new opj_image_cmptparm_t[nrComponents];
  for (int cnt = 0; cnt < nrComponents; cnt++)
  {
    componentParameters[cnt].dx = encodeParameters.subsampling_dx;
    componentParameters[cnt].dy = encodeParameters.subsampling_dy;
    componentParameters[cnt].h = tileSize;
    componentParameters[cnt].w = tileSize;
    componentParameters[cnt].prec = depth;
    componentParameters[cnt].bpp = depth;
    componentParameters[cnt].sgnd = 0;
    componentParameters[cnt].x0 = 0;
    componentParameters[cnt].y0 = 0;
  }
  // Also set the colorspace
  OPJ_COLOR_SPACE jpegColorSpace = OPJ_CLRSPC_GRAY;// Set the default.
  if (colorSpace == pathology::ColorType::RGB || colorSpace == pathology::ColorType::ARGB) {
    jpegColorSpace = OPJ_CLRSPC_SRGB;
  }

  // Get a J2K compressor handle.
  opj_codec_t* encoder = opj_create_compress(OPJ_CODEC_J2K);

  //Catch events using our callbacks and give a local context.
  opj_set_info_handler(encoder, info_callback, NULL);
  opj_set_warning_handler(encoder, warning_callback, NULL);
  opj_set_error_handler(encoder, error_callback, NULL);

  //Set the "OpenJpeg like" stream data.
  opj_memory_stream encodingBuffer;
  encodingBuffer.data = (OPJ_UINT8 *) new char[size];
  encodingBuffer.size = size;
  encodingBuffer.offset = 0;

  //Create an image struct.
  opj_image* encodedImage = opj_image_create(nrComponents, &componentParameters[0], jpegColorSpace);

  // Set image offset and reference grid
  encodedImage->x0 = 0;
  encodedImage->y0 = 0;
  encodedImage->x1 = (OPJ_UINT32)(componentParameters[0].w - 1) *	(OPJ_UINT32)encodeParameters.subsampling_dx + 1;
  encodedImage->y1 = (OPJ_UINT32)(componentParameters[0].h - 1) * (OPJ_UINT32)encodeParameters.subsampling_dy + 1;

  // Setup the encoder parameters using the current image and user parameters.
  OPJ_BOOL setupSuccess = opj_setup_encoder(encoder, &encodeParameters, encodedImage);
 
  //(Re)set the buffer pointerss.
  OPJ_INT32 **componentBuffer_ptr = new OPJ_INT32*[encodedImage->numcomps];
  for (int cnt = 0; cnt < encodedImage->numcomps; cnt++) {
    componentBuffer_ptr[cnt] = (OPJ_INT32*)(encodedImage->comps[cnt].data);
  }

  // Open a byte stream for writing.
  opj_stream_t * l_stream = opj_stream_create_default_memory_stream(&encodingBuffer, OPJ_FALSE);
  unsigned int bpc = (componentParameters[0].bpp + 7) / 8 ;
  //Set the color stuff.
  unsigned char* movingDataPointer = (unsigned char*)data;
  for (int index = 0; index < int(size / (encodedImage->numcomps * bpc)); index++) {
    for (int cmp = 0; cmp < nrComponents; cmp++) {
      for (int byteCnt = 0; byteCnt < bpc; byteCnt++)
      {
        *componentBuffer_ptr[cmp] |= (((OPJ_INT32)*movingDataPointer & 0xFF) << (8 * byteCnt));
        movingDataPointer++;
      }
      *componentBuffer_ptr[cmp]++;
    }
  }

  //Filled the buffers and ready to go.
  //encode the image.
  OPJ_BOOL startEncodeSuccess = opj_start_compress(encoder, encodedImage, l_stream);
  OPJ_BOOL encodeSuccess = opj_encode(encoder, l_stream);
  OPJ_BOOL endEncodeSuccess = opj_end_compress(encoder, l_stream);

  //Done with not reusables.
  opj_stream_destroy(l_stream);
  opj_image_destroy(encodedImage);
  opj_destroy_codec(encoder);
  
  // Change to encoded size 
  size = encodingBuffer.offset + 1;

  unsigned char * enc_bytes_ptr = encodingBuffer.data; // Pointer to start of encoded data
  movingDataPointer = (unsigned char*)data;
  for (int cnt = 0; cnt <= encodingBuffer.offset; cnt++) {
    *movingDataPointer++ = *enc_bytes_ptr++;
  }
  delete[] encodingBuffer.data;
  delete[] componentParameters;
}