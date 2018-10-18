#include "IconCreation.h"

#include <functional>
#include <stdexcept>

#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP::Worklist::GUI
{
	void CreateIcons(const DataTable& image_items, QStandardItemModel* image_model, QListView* image_view, const size_t size)
	{
		QList<QStandardItem*> items_to_insert;
		for (size_t item = 0; item < image_items.Size(); ++item)
		{
			std::vector<const std::string*> record(image_items.At(item, { "id", "location", "title" }));

			try
			{
				items_to_insert.push_back(new QStandardItem(CreateIcon(*record[1], size), QString(record[2]->data())));
				items_to_insert.back()->setData(QVariant(QString(record[1]->data())));
			}
			catch (const std::exception& e)
			{
				// Ignore icon creation.
			}
		}
		image_model->insertColumn(0, items_to_insert);

		// TODO: Figure out why async view / model updating doesn't seem to refresh the image_view.
		image_view->resize(image_view->width() - 1, image_view->height());
		image_view->resize(image_view->width() + 1, image_view->height());
	}

	QIcon CreateIcon(const std::string& filepath, const size_t size)
	{
		MultiResolutionImageReader reader;
		std::unique_ptr<MultiResolutionImage> image(reader.open(filepath));
			
		if (image)
		{
			unsigned char* data(nullptr);
			std::vector<unsigned long long> dimensions;

			if (image->getNumberOfLevels() > 1)
			{
				dimensions = image->getLevelDimensions(image->getNumberOfLevels() - 1);
				image->getRawRegion(0, 0, dimensions[0], dimensions[1], image->getNumberOfLevels() - 1, data);
			}
			else
			{
				dimensions = image->getDimensions();
				image->getRawRegion(0, 0, dimensions[0], dimensions[1], 0, data);
			}

			QImage image(dimensions[0], dimensions[1], QImage::Format::Format_BGR30);
			size_t base_index = 0;
			for (size_t y = 0; y < dimensions[1]; ++y)
			{
				for (size_t x = 0; x < dimensions[0]; ++x)
				{
					image.setPixel(x, y, qRgb(data[base_index + 2], data[base_index + 1], data[base_index]));
					base_index += 3;
				}
			}

			delete data;
			QPixmap pixmap(QPixmap::fromImage(image).scaled(QSize(size, size), Qt::AspectRatioMode::KeepAspectRatio));
			return QIcon(pixmap);
		}
		else
		{
			throw std::runtime_error("Unable to read image: " + filepath);
		}
	}
}