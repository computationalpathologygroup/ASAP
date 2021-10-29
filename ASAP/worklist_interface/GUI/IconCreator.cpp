#include "IconCreator.h"

#include <QStandardPaths>

#include <functional>
#include <stdexcept>

#include <multiresolutionimageinterface/MultiResolutionImage.h>
#include <multiresolutionimageinterface/MultiResolutionImageReader.h>
#include <multiresolutionimageinterface/MultiResolutionImageFactory.h>

namespace ASAP
{
	int IconCreator::m_icon_size = 200;

	IconCreator::IconCreator(void) : m_placeholder_icon(IconCreator::createBlankIcon()), m_invalid_icon(IconCreator::createInvalidIcon())
	{
		m_thumbnail_cache = new ThumbnailCache(QStandardPaths::writableLocation(QStandardPaths::StandardLocation::CacheLocation));		
	}

	IconCreator::~IconCreator() {
		if (m_thumbnail_cache) {
			delete m_thumbnail_cache;
		}
	}

	bool IconCreator::insertIcon(const std::pair<int, std::string>& index_location)
	{
		bool isValid = false;
		QIcon itemIcon = createIcon(index_location.second, IconCreator::m_icon_size);	
		if (itemIcon.isNull()) {
			itemIcon = m_invalid_icon;
		}
		else
		{
			isValid = true;
		}
			
		// Signals the model that the item has had a certain amount of icon changes.
		requiresItemRefresh(index_location.first, itemIcon);		
		return isValid;
	}

	QIcon IconCreator::createIcon(const std::string& filepath, const size_t size)
	{
		QImage scaled_image = m_thumbnail_cache->getThumbnailFromCache(QString::fromStdString(filepath));
		if (scaled_image.isNull()) {
			MultiResolutionImageReader reader;
			std::unique_ptr<MultiResolutionImage> image(reader.open(filepath));

			if (image)
			{
				unsigned char* data(nullptr);
				std::vector<unsigned long long> dimensions;

				if (image->getDataType() == pathology::UChar) {
					if (image->getNumberOfLevels() > 1)
					{
						dimensions = image->getLevelDimensions(image->getNumberOfLevels() - 1);
						image->getRawRegion(0, 0, dimensions[0], dimensions[1], image->getNumberOfLevels() - 1, data);
					}
					else
					{
						dimensions = image->getDimensions();
						if (dimensions[0] * dimensions[1] < (1024 * 1024)) {
							image->getRawRegion(0, 0, dimensions[0], dimensions[1], 0, data);
						}
						else {
							return m_invalid_icon;
						}
					}
				}
				else {
					return m_invalid_icon;
				}

				// Gets the largest dimension and creates an offset for the smallest.
				unsigned long long max_dim = std::max(dimensions[0], dimensions[1]);
				size_t offset_x = dimensions[0] == max_dim ? 0 : (dimensions[1] - dimensions[0]) / 2;
				size_t offset_y = dimensions[1] == max_dim ? 0 : (dimensions[0] - dimensions[1]) / 2;

				QImage qimage(max_dim, max_dim, QImage::Format::Format_RGB888);
				qimage.fill(Qt::white);

				// Writes the multiresolution pixel data into the image.
				size_t base_index = 0;
				if (image->getColorType() == pathology::ColorType::Monochrome) {
					for (size_t y = 0; y < dimensions[1]; ++y)
					{
						for (size_t x = 0; x < dimensions[0]; ++x)
						{
							qimage.setPixel(x + offset_x, y + offset_y, qRgb(data[base_index], data[base_index], data[base_index]));
							base_index += 1;
						}
					}
				}
				else if (image->getColorType() == pathology::ColorType::RGB) {
					for (size_t y = 0; y < dimensions[1]; ++y)
					{
						for (size_t x = 0; x < dimensions[0]; ++x)
						{
							qimage.setPixel(x + offset_x, y + offset_y, qRgb(data[base_index], data[base_index + 1], data[base_index + 2]));
							base_index += 3;
						}
					}
				}
				else if (image->getColorType() == pathology::ColorType::RGBA) {
					for (size_t y = 0; y < dimensions[1]; ++y)
					{
						for (size_t x = 0; x < dimensions[0]; ++x)
						{
							qimage.setPixel(x + offset_x, y + offset_y, qRgb(data[base_index], data[base_index + 1], data[base_index + 2]));
							base_index += 4;
						}
					}
				}

				delete[] data;
				scaled_image = qimage.scaled(QSize(size, size), Qt::AspectRatioMode::KeepAspectRatio);
				m_thumbnail_cache->addThumbnailToCache(QString::fromStdString(filepath), scaled_image);
				QPixmap pixmap(QPixmap::fromImage(scaled_image));
				return QIcon(pixmap);
			}
			else
			{
				return QIcon();
			}
		}
		else {
			QPixmap pixmap(QPixmap::fromImage(scaled_image));
			return QIcon(pixmap);
		}
	}

	QIcon IconCreator::createBlankIcon()
	{
		QImage image(IconCreator::m_icon_size, IconCreator::m_icon_size, QImage::Format::Format_BGR30);
		image.fill(Qt::white);
		return QIcon(QPixmap::fromImage(image));
	}

	QIcon IconCreator::createInvalidIcon()
	{
		QPixmap invalid_icon(":/ASAP_Worklist_icons/unavailable.png");
		return QIcon(invalid_icon.scaled(IconCreator::m_icon_size, IconCreator::m_icon_size, Qt::AspectRatioMode::KeepAspectRatio));
	}
}