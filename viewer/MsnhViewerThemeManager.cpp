#include "MsnhViewerThemeManager.h"

#include <QDebug>
#include <QFile>
#include <QByteArray>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>

namespace MsnhViewer
{
    MsnhViewer::ThemeManager& ThemeManager::instance()
    {
        static ThemeManager instance;
        return instance;
    }


    NodeTheme ThemeManager::getNodeTheme() const
    {
        return nodeTheme_;
    }


    AttributeTheme ThemeManager::getAttributeTheme() const
    {
        return attributeTheme_;
    }


    DataTypeTheme ThemeManager::getDataTypeTheme(QString const& dataType)
    {
        if (dataTypeThemes_.contains(dataType))
        {
            return dataTypeThemes_[dataType];
        }
        return dataTypeThemes_["default"];
    }


    bool ThemeManager::load(const QString& themeFilename)
    {
        QFile io(themeFilename);
        if (!io.open(QIODevice::ReadOnly))
        {
            qWarning() << "Can't open theme file" << themeFilename;
            return false;
        }
        QByteArray file_data = io.readAll();
        QJsonParseError error;
        QJsonDocument theme_document = QJsonDocument::fromJson(file_data, &error);
        if (error.error != QJsonParseError::NoError)
        {
            qWarning() << "Error while parsing theme file:" << error.errorString();
            return false;
        }

        if (!parseNode(theme_document.object()))
        {
            return false;
        }

        if (!parseAttribute(theme_document.object()))
        {
            return false;
        }

        if (!parseDataType(theme_document.object()))
        {
            return false;
        }

        return true;
    }


    bool ThemeManager::parseNode(QJsonObject const& json)
    {
        if (!(json.contains("node")&&json["node"].isObject()))
        {
            qWarning() << "Can't parse node";
            return false;
        }
        QJsonObject node = json["node"].toObject();

        QJsonObject name = node["name"].toObject();
        nodeTheme_.name_font  = parseFont(name["font"].toObject());
        nodeTheme_.name_color = parseColor(name["font"].toObject());

        nodeTheme_.background = parseColor(node["background"].toObject());

        QJsonObject border = node["border"].toObject();
        nodeTheme_.border = parseColor(border["normal"].toObject());
        nodeTheme_.border_selected = parseColor(border["selected"].toObject());;

        QJsonObject type = node["type"].toObject();
        nodeTheme_.type_font  = parseFont(type["font"].toObject());
        nodeTheme_.type_color = parseColor(type["font"].toObject());
        nodeTheme_.type_background = parseColor(type["background"].toObject());;

        return true;
    }


    bool ThemeManager::parseAttribute(QJsonObject const& json)
    {
        if (!(json.contains("attribute")&&json["attribute"].isObject()))
        {
            qWarning() << "Can't parse attribute";
            return false;
        }
        QJsonObject attribute = json["attribute"].toObject();

        attributeTheme_.background = parseColor(attribute["background"].toObject());
        attributeTheme_.background_alt = parseColor(attribute["background_alt"].toObject());

        auto parseMode = [this](QJsonObject const& json, AttributeTheme::Mode& mode)
        {
            mode.font = parseFont(json["font"].toObject());
            mode.font_color = parseColor(json["font"].toObject());

            QJsonObject connector = json["connector"].toObject();
            mode.connector.border_color = parseColor(json["connector"].toObject());
            mode.connector.border_width = connector["width"].toInt();
        };

        parseMode(attribute["minimize"].toObject(),  attributeTheme_.minimize);
        parseMode(attribute["normal"].toObject(),    attributeTheme_.normal);
        parseMode(attribute["highlight"].toObject(), attributeTheme_.highlight);

        return true;
    }


    bool ThemeManager::parseDataType(QJsonObject const& json)
    {
        if (!(json.contains("data_type")&&json["data_type"].isObject()))
        {
            qWarning() << "Can't parse data type";
            return false;
        }
        QJsonObject data_type = json["data_type"].toObject();

        DataTypeTheme theme;
        theme.enable = parseColor(data_type["enable_default"].toObject());
        theme.disable = parseColor(data_type["disable"].toObject());
        theme.neutral = parseColor(data_type["neutral"].toObject());
        dataTypeThemes_["default"] = theme;

        QJsonObject enable_custom = data_type["enable_custom"].toObject();
        for (auto const& key : enable_custom.keys())
        {
            theme.enable = parseColor(enable_custom[key].toObject());
            dataTypeThemes_[key] = theme;
        }

        return true;
    }


    QColor ThemeManager::parseColor(QJsonObject const& json)
    {
        QJsonArray rgba = json["rgba"].toArray();
        return QColor{ rgba[0].toInt(), rgba[1].toInt(), rgba[2].toInt(), rgba[3].toInt()};
    }


    QFont ThemeManager::parseFont(QJsonObject const& json)
    {
        QString name = json["name"].toString();
        QString weight = json["weight"].toString();
        int size = json["size"].toInt();

        QFont::Weight qweight = QFont::Normal;
        if (weight == "Bold")
        {
            qweight = QFont::Bold;
        }
        else if (weight == "Medium")
        {
            qweight = QFont::Medium;
        }
        else if (weight == "Normal")
        {
            qweight = QFont::Normal;
        }
        else if (weight == "Light")
        {
            qweight = QFont::Light;
        }
        else
        {
            qWarning() << "Unknown font weight" << weight;
        }

        return QFont{ name, size, qweight };
    }

}
