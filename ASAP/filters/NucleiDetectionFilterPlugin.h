#ifndef _NucleiDetectionFilterPluginH
#define _NucleiDetectionFilterPluginH

#include "interfaces/interfaces.h"

template <class inType> 
class NucleiDetectionFilter;

class QDialog;

class NucleiDetectionFilterPlugin : public ImageFilterPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.NucleiDetectionPlugin/1.0")
    Q_INTERFACES(ImageFilterPluginInterface)

public :
    NucleiDetectionFilterPlugin();
    NucleiDetectionFilterPlugin(const NucleiDetectionFilterPlugin& other);

    QString name() const;
    bool initialize(const ImageSource* img);
    void filter(const Patch<double> &input, QVariant &output);
    QPointer<QWidget> getSettingsPanel();
    ImageFilterPluginInterface* clone() const;
    QIcon icon() const;

private :

    void updateSettingsPanelFromFilter();
    void initializeSettingsPanel();
    bool _monochromeInput;
    
private slots :
    void updateFilterFromSettingsPanel();
    void revertStainToDefault();

signals:
    void filterParametersChanged();

};

#endif
