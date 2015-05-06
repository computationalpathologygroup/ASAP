#pragma once

// If the image file is truncated, we insert EOI marker to signal to the library to stop processing and display the image instead of raising an error.
#define PROCESS_TRUNCATED_IMAGES

static void mem_init_source( j_decompress_ptr cinfo ) 
{
}

static void mem_term_source( j_decompress_ptr cinfo ) 
{
}

static boolean mem_fill_input_buffer( j_decompress_ptr cinfo )
{
#ifdef PROCESS_TRUNCATED_IMAGES
    jpeg_source_mgr* src = cinfo->src;

	static const JOCTET EOI_BUFFER[ 2 ] = { (JOCTET)0xFF, (JOCTET)JPEG_EOI };
    src->next_input_byte = EOI_BUFFER;
    src->bytes_in_buffer = sizeof( EOI_BUFFER );
#else
	ERREXIT( cinfo, JERR_INPUT_EOF );
#endif
	return TRUE;
}

static void mem_skip_input_data( j_decompress_ptr cinfo, long num_bytes )
{
    jpeg_source_mgr* src = (jpeg_source_mgr*)cinfo->src;

    if ( 1 > num_bytes )
		return;

	if ( num_bytes < src->bytes_in_buffer )
	{
		src->next_input_byte += (size_t)num_bytes;
		src->bytes_in_buffer -= (size_t)num_bytes;
	}
	else
	{
#ifdef PROCESS_TRUNCATED_IMAGES
		src->bytes_in_buffer = 0;
#else
		ERREXIT( cinfo, JERR_INPUT_EOF );
#endif
	}
}

static void jpeg_mem_src( j_decompress_ptr cinfo, jpeg_source_mgr * const src, void const * const buffer, long nbytes )
{
    src->init_source = mem_init_source;
    src->fill_input_buffer = mem_fill_input_buffer;
    src->skip_input_data = mem_skip_input_data;
    src->resync_to_restart = jpeg_resync_to_restart;
    src->term_source = mem_term_source;
    src->bytes_in_buffer = nbytes;
    src->next_input_byte = (JOCTET*)buffer;
    cinfo->src = src;
}