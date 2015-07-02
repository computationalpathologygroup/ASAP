#include "JPEG2000Codec.h"
// Include LIBJASPER for JPEG2000 compression
#include "jasper/jasper.h"
#include <string>
#include <vector>
#include <sstream>

using namespace std;

JPEG2000Codec::JPEG2000Codec() {
  jas_init();
}

JPEG2000Codec::~JPEG2000Codec() {
  jas_cleanup();
}

void JPEG2000Codec::decode(char* buf, const unsigned int& inSize, const unsigned int& outSize) const
{
    jas_image_t *image = NULL;
    jas_stream_t *jpcstream = NULL;
    char *opts=0;
    std::vector<jas_matrix_t*> components;

    jpcstream = jas_stream_memopen(buf, inSize);
    image=jpc_decode(jpcstream, opts);
    if (image) {
      for (int i = 0; i < image->numcmpts_; ++i) {
        jas_matrix_t* tmp = jas_matrix_create(jas_image_height(image), jas_image_width(image));        
        jas_image_readcmpt(image, i, 0, 0, jas_image_width(image), jas_image_height(image), tmp);
        components.push_back(tmp);
      }
      for (int j = 0; j < components.size(); ++j) {
        jas_matrix_t* cmp = components[j];
        for (int i = j; i < outSize; i += image->numcmpts_) {
          buf[i] = cmp->data_[i / image->numcmpts_];
        }
      }
    }
    for (std::vector<jas_matrix_t*>::iterator it = components.begin(); it != components.end(); ++it) {
      jas_matrix_destroy(*it);
    }
    components.clear();
    jas_stream_close(jpcstream);
    jas_image_destroy(image);
}

void JPEG2000Codec::encode(char* data, unsigned int& size, const unsigned int& tileSize, const unsigned int& depth, const unsigned int& nrComponents, float& rate) const
{
    jas_image_cmptparm_t* component_info = new jas_image_cmptparm_t[4];
    for( int i = 0; i < nrComponents; i++ )
    {
        component_info[i].tlx = 0;
        component_info[i].tly = 0;
        component_info[i].hstep = 1;
        component_info[i].vstep = 1;
        component_info[i].width = tileSize;
        component_info[i].height = tileSize;
        component_info[i].prec = depth;
        component_info[i].sgnd = 0;
    }
    jas_image_t *img = jas_image_create(nrComponents, component_info, (nrComponents == 1 || nrComponents == 2) ? JAS_CLRSPC_SGRAY : JAS_CLRSPC_SRGB);
    if(nrComponents == 1) {
      jas_image_setcmpttype(img, 0, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_GRAY_Y));
    }
    else if(nrComponents == 2) {
      jas_image_setcmpttype(img, 0, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_GRAY_Y));
      jas_image_setcmpttype(img, 1, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_GRAY_Y));
    }
    else if (nrComponents == 3)
    {
      jas_image_setcmpttype(img, 0, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_R));
      jas_image_setcmpttype(img, 1, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_G));
      jas_image_setcmpttype(img, 2, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_B));
    }
    else
    {
      jas_image_setcmpttype(img, 0, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_R));
      jas_image_setcmpttype(img, 1, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_G));
      jas_image_setcmpttype(img, 2, JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_B));
      jas_image_setcmpttype(img, 3, JAS_IMAGE_CT_COLOR(JAS_IMAGE_CT_OPACITY));
    }
    jas_matrix_t *row = jas_matrix_create( 1, tileSize );
    for( int y = 0; y < tileSize; ++y)
    {
        unsigned char* tmp = (unsigned char*)data + tileSize*y*nrComponents;
        for( int i = 0; i < nrComponents; i++ )
        {
            for( int x = 0; x < tileSize; x++) {
                jas_matrix_setv( row, x, tmp[x * nrComponents + i] );
            }
            jas_image_writecmpt( img, i, 0, y, tileSize, 1, row );
        }
    }
    jas_matrix_destroy( row );
    jas_stream_t *stream = jas_stream_memopen(data, size);    
    if (rate<0.01) {
      rate = 0.01;
    } else if (rate > 1.0) {
      rate = 1.0;
    }
    if (rate == 1.0) {
      string opts = std::string("mode=int");
      jas_image_encode( img, stream, jas_image_strtofmt( (char*)"jpc" ), (char*)opts.c_str());
    } else {
      std::stringstream ssm;
      ssm << rate;
      std::string rateString;
      ssm >> rateString;
      string opts = std::string("rate=") + rateString;
      jas_image_encode( img, stream, jas_image_strtofmt( (char*)"jpc" ), (char*)opts.c_str());
    }
    size = stream->rwcnt_;
    jas_stream_close( stream );
    jas_image_destroy( img );
}