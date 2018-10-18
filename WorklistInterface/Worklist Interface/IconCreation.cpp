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

	IconCreator::~IconCreator(void)
	{
		m_next_message_access_.lock();
		m_next_message_access_.unlock();
	}

	void IconCreator::InsertIcons(const DataTable& image_items, QStandardItemModel* image_model, QStatusBar* status_bar, const size_t size)
	{
		// Stops the message bar from being overriden during a message writing.
		connect(status_bar,
				&QStatusBar::messageChanged,
				this,
				&IconCreator::OnMessageChanged_);

		// TODO: Create placeholders

		// Fills the placerholders
		QString total_size(std::to_string(image_items.Size()).data());
		for (size_t item = 0; item < image_items.Size(); ++item)
		{
			m_next_message_access_.lock();
			m_next_message_ = "Loading " + QString(std::to_string(item).data()) + "out of " + total_size;
			m_next_message_access_.unlock();			

			std::vector<const std::string*> record(image_items.At(item, { "id", "location", "title" }));
			try
			{
				QStandardItem* standard_item(new QStandardItem(CreateIcon_(*record[1], size), QString(record[2]->data())));
				standard_item->setData(QVariant(QString(record[1]->data())));

				image_model->setRowCount(image_model->rowCount() + 1);
				image_model->setItem(image_model->rowCount() - 1, 0, standard_item);
			}
			catch (const std::exception& e)
			{
				// Ignore icon creation.
			}
		}

		m_next_message_.clear();
		m_status_bar_ = nullptr;
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

	void IconCreator::OnMessageChanged_(const QString& text)
	{
		m_next_message_access_.lock();
		if (text != m_next_message_)
		{
			m_status_bar_->showMessage(m_next_message_);
		}
		m_next_message_access_.unlock();
	}
}