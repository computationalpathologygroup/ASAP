#include "JPEG2000Codec.h"
// Include LIBJASPER for JPEG2000 compression
#include "jasper/jasper.h"
#include <string>
#include <sstream>

using namespace std;

JPEG2000Codec::JPEG2000Codec() {
  jas_init();
}

JPEG2000Codec::~JPEG2000Codec() {
  jas_cleanup();
}

void JPEG2000Codec::decode(char* buf, const unsigned int& size) const
{
      jas_image_t *image=0;
      jas_stream_t *jpcstream;
      jas_image_cmpt_t **comps;
      char *opts=0;
      jas_matrix_t *red, *green, *blue;

      jpcstream=jas_stream_memopen(buf,size);
      image=jpc_decode(jpcstream,opts);
      red=jas_matrix_create(jas_image_height(image), jas_image_width(image));
      green=jas_matrix_create(jas_image_height(image), jas_image_width(image));
      blue=jas_matrix_create(jas_image_height(image), jas_image_width(image));
      jas_image_readcmpt(image,0,0,0,jas_image_width(image),
                         jas_image_height(image),red);
      jas_image_readcmpt(image,1,0,0,jas_image_width(image),
                         jas_image_height(image),green);
      jas_image_readcmpt(image,2,0,0,jas_image_width(image),
                         jas_image_height(image),blue);

      for (int i = 0; i < size; i+=3) {
        buf[i] = red->data_[i/3];
        buf[i+1] = green->data_[i/3];
        buf[i+2] = blue->data_[i/3];
      }

      jas_matrix_destroy(red);
      jas_matrix_destroy(green);
      jas_matrix_destroy(blue);
      jas_stream_close(jpcstream);
      jas_image_destroy(image);
}

void JPEG2000Codec::encode(char* data, unsigned int& size, const unsigned int& tileSize, const unsigned int& depth, const unsigned int& nrComponents, float& rate) const
{
    jas_image_cmptparm_t component_info[4];
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
    jas_image_t *img = jas_image_create( nrComponents, component_info, (nrComponents == 1) ? JAS_CLRSPC_SGRAY : JAS_CLRSPC_SRGB );
    if(nrComponents == 1) {
        jas_image_setcmpttype( img, 0, JAS_IMAGE_CT_GRAY_Y );
    }
    else if(nrComponents == 2) {
        jas_image_setcmpttype( img, 0, JAS_IMAGE_CT_GRAY_Y );
        jas_image_setcmpttype( img, 1, JAS_IMAGE_CT_GRAY_Y );
    }
    else if (nrComponents == 3)
    {
        jas_image_setcmpttype( img, 0, 2 );
        jas_image_setcmpttype( img, 1, 1 );
        jas_image_setcmpttype( img, 2, 0 );
    }
    else
    {
        jas_image_setcmpttype( img, 0, 3 );
        jas_image_setcmpttype( img, 1, 2 );
        jas_image_setcmpttype( img, 2, 1 );
        jas_image_setcmpttype( img, 2, 0 );
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
      string opts = std::string("rate=1.0 mode=int");
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