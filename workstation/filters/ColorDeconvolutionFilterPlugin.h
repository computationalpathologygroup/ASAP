#ifndef _ColorDeconvolutionFilterPluginH
#define _ColorDeconvolutionFilterPluginH

#include "interfaces/interfaces.h"

template <class inType>
class ColorDeconvolutionFilter;

class QDialog;

class ColorDeconvolutionFilterPlugin : public ImageFilterPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.ColorDeconvolutionFilter/1.0")
    Q_INTERFACES(ImageFilterPluginInterface)

public :
    ColorDeconvolutionFilterPlugin();
    ColorDeconvolutionFilterPlugin(const ColorDeconvolutionFilterPlugin& other);

    QString name() const;
    QIcon icon() const;
    bool initialize(const ImageSource* image);
    void filter(const Patch<double> &input, QVariant &output);
    QPointer<QWidget> getSettingsPanel();
    ImageFilterPluginInterface* clone() const;

private :
    void updateSettingsPanelFromFilter();
    void initializeSettingsPanel();

private slots :
    void updateFilterFromSettingsPanel();
    void revertStainToDefault();

signals:
    void filterParametersChanged();

};

#endif
