#include "IconCreation.h"

#include <functional>
#include <stdexcept>

#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP::Worklist::GUI
{
	IconCreator::IconCreator(void)
	{
	}

	void IconCreator::InsertIcons(const DataTable& image_items, QStandardItemModel* image_model, const size_t size)
	{
		// Creates placeholder items
		image_model->setRowCount(image_items.Size());
		std::vector<QStandardItem*> standard_items;
		for (size_t item = 0; item < image_items.Size(); ++item)
		{
			std::vector<const std::string*> record(image_items.At(item, { "id", "title" }));
			standard_items.push_back(new QStandardItem(CreateBlankIcon_(size), QString(record[1]->data())));
			standard_items.back()->setData(QVariant(QString(record[0]->data())));
			image_model->setItem(item, 0, standard_items.back());
		}

		// Fills the placeholders		
		QString total_size(std::to_string(image_items.Size()).data());
		for (size_t item = 0; item < image_items.Size(); ++item)
		{
			RequiresStatusBarChange("Loading " + QString::fromStdString(std::to_string(item)) + " out of " + total_size);
			try
			{
				std::vector<const std::string*> record(image_items.At(item, { "location", "title" }));
				standard_items[item]->setIcon(CreateIcon_(*record[1], size));
			}
			catch (const std::exception& e)
			{
				// Ignore icon creation.
			}
		}
		RequiresStatusBarChange("Finished loading thumbnails");
	}

	QIcon IconCreator::CreateIcon_(const std::string& filepath, const size_t size)
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

			// Gets the largest dimension and creates an offset for the smallest.
			unsigned long long max_dim	= std::max(dimensions[0], dimensions[1]);
			size_t offset_x				= dimensions[0] == max_dim ? 0 : (dimensions[1] - dimensions[0]) / 2;
			size_t offset_y				= dimensions[1] == max_dim ? 0 : (dimensions[0] - dimensions[1]) / 2;

			QImage image(max_dim, max_dim, QImage::Format::Format_BGR30);
			image.fill(Qt::white);

			// Writes the multiresolution pixel data into the image.
			size_t base_index = 0;
			for (size_t y = 0; y < dimensions[1]; ++y)
			{
				for (size_t x = 0; x < dimensions[0]; ++x)
				{
					image.setPixel(x + offset_x, y + offset_y, qRgb(data[base_index + 2], data[base_index + 1], data[base_index]));
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

	QIcon IconCreator::CreateBlankIcon_(const size_t size)
	{
		QImage image(size, size, QImage::Format::Format_BGR30);
		image.fill(Qt::white);
		return QIcon(QPixmap::fromImage(image));
	}
}